import calendar
import logging
import time
import datetime
from typing import List, Dict, Optional
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from oauth2client.service_account import ServiceAccountCredentials
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class BCPScraper:
    """Clase para scraping de tablas de cotizaciones del BCP con paralelización."""
    
    # Mapeo de meses en español a inglés
    MONTHS_ES_TO_EN = {
        "ENE": "JAN",
        "FEB": "FEB",
        "MAR": "MAR",
        "ABR": "APR",
        "MAY": "MAY",
        "JUN": "JUN",
        "JUL": "JUL",
        "AGO": "AUG",
        "SEP": "SEP",
        "OCT": "OCT",
        "NOV": "NOV",
        "DIC": "DEC"
    }
    
    def __init__(self, config: Dict):
        """Inicializa el scraper con configuración."""
        self.config = config
        self.driver_pool = [self._setup_driver() for _ in range(3)]  # Pool de 3 drivers
        self.wait_timeout = config.get('wait_timeout', 30)
        
    def _setup_driver(self) -> webdriver.Edge:
        """Configura el WebDriver con ventana de 1200x900 px."""
        options = Options()
        options.add_argument("--window-size=1200,900")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        service = Service()
        return webdriver.Edge(service=service, options=options)
    
    def _get_dropdown_options(self, driver: webdriver.Edge, element_name: str) -> Dict[str, str]:
        """Obtiene las opciones disponibles en un dropdown como {texto: valor}."""
        try:
            select_element = WebDriverWait(driver, self.wait_timeout).until(
                EC.presence_of_element_located((By.NAME, element_name)))
            select = Select(select_element)
            return {option.text.strip(): option.get_attribute("value") for option in select.options}
        except Exception as e:
            logger.error(f"Error al obtener opciones de {element_name}: {str(e)}")
            return {}
    
    def _get_available_years(self, driver: webdriver.Edge) -> List[str]:
        """Obtiene todos los años disponibles en el dropdown."""
        driver.get(self.config['url'])
        time.sleep(2)
        options = self._get_dropdown_options(driver, "anho")
        return [v for k, v in options.items() if v]  # Devuelve los valores no vacíos
    
    def _get_available_currencies(self, driver: webdriver.Edge) -> List[str]:
        """Obtiene todas las monedas disponibles en el dropdown."""
        driver.get(self.config['url'])
        time.sleep(2)
        options = self._get_dropdown_options(driver, "moneda")
        return [v for k, v in options.items() if v]  # Devuelve los valores no vacíos
    
    def _select_dropdown(self, driver: webdriver.Edge, element_name: str, target: str) -> bool:
        """Selecciona una opción en un dropdown por texto visible o valor."""
        try:
            options = self._get_dropdown_options(driver, element_name)
            if not options:
                return False
                
            # Buscar coincidencia exacta o parcial
            for text, value in options.items():
                if target in text or target == value:
                    select_element = WebDriverWait(driver, self.wait_timeout).until(
                        EC.element_to_be_clickable((By.NAME, element_name)))
                    select = Select(select_element)
                    
                    try:
                        select.select_by_visible_text(text)
                    except:
                        select.select_by_value(value)
                    
                    time.sleep(1)
                    return True
            
            logger.error(f"No se encontró '{target}' en las opciones. Disponibles: {list(options.keys())}")
            return False
        except Exception as e:
            logger.error(f"Error al seleccionar {target} en {element_name}: {str(e)}")
            return False
    
    def _get_currency_name(self, driver: webdriver.Edge, currency_value: str) -> str:
        """Obtiene el nombre legible de la moneda."""
        options = self._get_dropdown_options(driver, "moneda")
        return next((text for text, val in options.items() if val == currency_value), currency_value)
    
    def _parse_table_data(self, table_element) -> List[List]:
        """Extrae y parsea los datos de la tabla."""
        data = []
        rows = table_element.find_elements(By.TAG_NAME, "tr")
        
        if len(rows) != 32:
            logger.warning(f"Tabla no tiene 32 filas, tiene {len(rows)}")
            return data
        
        for row in rows[1:32]:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                day_th = row.find_element(By.TAG_NAME, "th")
                
                if not day_th.text.strip().isdigit():
                    continue
                    
                day = int(day_th.text.strip())
                row_data = [day]
                
                for cell in cells[:12]:
                    val = cell.text.strip()
                    
                    if not val or val.upper() in ("ND", "N/D", "NA", "N/A"):
                        row_data.append(None)
                    else:
                        try:
                            cleaned = val.replace(".", "").replace(",", ".")
                            if cleaned.replace(".", "").isdigit():
                                row_data.append(float(cleaned))
                            else:
                                row_data.append(None)
                        except ValueError:
                            row_data.append(None)
                
                data.append(row_data)
            except Exception:
                continue
        
        return data
    
    def _scrape_combination(self, driver: webdriver.Edge, year: str, currency: str) -> Optional[pd.DataFrame]:
        """Función para scrapear una combinación específica de año y moneda."""
        try:
            logger.info(f"Procesando {year}/{currency} en hilo {id(driver)}")
            
            driver.get(self.config['url'])
            time.sleep(3)
            
            if not all([
                self._select_dropdown(driver, "anho", year),
                self._select_dropdown(driver, "moneda", currency)
            ]):
                return None
            
            submit_button = WebDriverWait(driver, self.wait_timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
            submit_button.click()
            
            WebDriverWait(driver, self.wait_timeout).until(
                lambda d: len(d.find_elements(By.ID, "cotizacion-interbancaria")) >= 2)
            time.sleep(2)
            
            tables = driver.find_elements(By.ID, "cotizacion-interbancaria")
            if len(tables) < 2:
                return None
            
            table_data = self._parse_table_data(tables[1])
            if not table_data:
                return None
            
            columns = ["DIA"] + list(self.MONTHS_ES_TO_EN.keys())
            df = pd.DataFrame(table_data, columns=columns)
            df['anho'] = year
            df['moneda'] = self._get_currency_name(driver, currency)
            
            return df
            
        except Exception as e:
            logger.error(f"Error procesando {year}/{currency}: {str(e)}")
            return None
    
    def scrape_data(self) -> List[pd.DataFrame]:
        """Extrae datos con paralelización (hasta 3 ventanas simultáneas)."""
        # Usar el primer driver para obtener la configuración
        main_driver = self.driver_pool[0]
        
        # Determinar años a procesar
        if self.config['years'] is None:
            # Si es None, obtener todos los años disponibles
            years_to_process = self._get_available_years(main_driver)
        elif not self.config['years']:
            # Si es lista vacía, también obtener todos los años
            years_to_process = self._get_available_years(main_driver)
        else:
            # Usar los años especificados en la configuración
            years_to_process = self.config['years']
        
        # Determinar monedas a procesar (si la lista está vacía, obtener todas)
        currencies_to_process = self.config['currencies'] if self.config['currencies'] else self._get_available_currencies(main_driver)
        
        logger.info(f"Años a procesar: {years_to_process}")
        logger.info(f"Monedas a procesar: {currencies_to_process}")
        
        data_frames = []
        combinations = [(year, currency) for year in years_to_process for currency in currencies_to_process]
        
        # Usar ThreadPoolExecutor para paralelizar el scraping
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # Asignar tareas a los drivers en round-robin
            for i, (year, currency) in enumerate(combinations):
                driver = self.driver_pool[i % 3]
                futures.append(executor.submit(self._scrape_combination, driver, year, currency))
            
            # Recoger resultados conforme se completan
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    data_frames.append(result)
        
        return data_frames
    
    def close(self):
        """Cierra todos los navegadores en el pool."""
        for driver in self.driver_pool:
            try:
                driver.quit()
            except:
                pass


class DataProcessor:
    """Procesa los datos extraídos."""
    
    @staticmethod
    def process_data(raw_data: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Transforma datos brutos a formato para exportación."""
        if not raw_data:
            return None
            
        df = pd.concat([df for df in raw_data if not df.empty], ignore_index=True)
        
        # Transformar a formato largo usando los meses en español
        df = df.melt(
            id_vars=["DIA", "anho", "moneda"],
            value_vars=list(BCPScraper.MONTHS_ES_TO_EN.keys()),
            var_name="MES_ES",
            value_name="Cotizacion"
        )
        
        # Convertir meses español a inglés para calendar
        df['MES_EN'] = df['MES_ES'].map(BCPScraper.MONTHS_ES_TO_EN)
        
        # Mapear nombres de mes en inglés a números (1-12)
        mes_map = {v.upper(): k for k, v in enumerate(calendar.month_abbr) if v}
        df['MES'] = df['MES_EN'].map(mes_map)
        
        # Crear columna de fecha en formato yyyy-mm-dd
        df['Fecha'] = pd.to_datetime(
            df[['anho', 'MES', 'DIA']].rename(columns={'anho': 'year', 'MES': 'month', 'DIA': 'day'}),
            errors='coerce'
        ).dt.strftime('%Y-%m-%d')  # Formato yyyy-mm-dd
        
        result = df[['moneda', 'Fecha', 'Cotizacion']]
        result.columns = ["Moneda", "Fecha", "Cotizacion"]
        
        return result.dropna(subset=['Cotizacion']).sort_values(['Moneda', 'Fecha'])


class GoogleSheetsUpdater:
    """Maneja la actualización incremental de Google Sheets."""
    
    def __init__(self, credentials_path: str):
        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        self.creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, self.scope)
        self.client = gspread.authorize(self.creds)
    
    def update_sheet(self, new_data: pd.DataFrame, spreadsheet_url: str, sheet_name: str, years: List[str], currencies: List[str]) -> bool:
        """Actualiza la hoja de cálculo conservando datos existentes."""
        try:
            spreadsheet = self.client.open_by_url(spreadsheet_url)
            
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
            except gspread.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=max(100, len(new_data)+1), cols=3)
                # Si es nueva, simplemente añadimos todos los datos
                set_with_dataframe(worksheet, new_data)
                return True
            
            # Obtener datos existentes
            existing_data = get_as_dataframe(worksheet, evaluate_formulas=True)
            
            if existing_data is None or existing_data.empty:
                set_with_dataframe(worksheet, new_data)
                return True
            
            # Si no se especificaron años/monedas (se obtuvieron todos), reemplazar todo
            if not years or not currencies:
                worksheet.clear()
                set_with_dataframe(worksheet, new_data)
                return True
            
            # Filtrar para eliminar solo los datos de los años y monedas que estamos actualizando
            mask = ~(
                existing_data['Moneda'].isin(currencies) & 
                existing_data['Fecha'].str[:4].isin(years)  # Extraer año de la fecha
            )
            data_to_keep = existing_data[mask]
            
            # Combinar con nuevos datos
            updated_data = pd.concat([data_to_keep, new_data], ignore_index=True)
            
            # Ordenar y limpiar
            updated_data = updated_data.sort_values(['Moneda', 'Fecha'])
            updated_data = updated_data.drop_duplicates(subset=['Moneda', 'Fecha'], keep='last')
            
            # Actualizar la hoja completa
            worksheet.clear()
            set_with_dataframe(worksheet, updated_data)
            
            return True
        except Exception as e:
            logger.error(f"Error al actualizar la hoja: {str(e)}")
            return False


def main():
    """Función principal."""
    config = {
        'url': "https://www.bcp.gov.py/webapps/web/cotizacion/monedas-historica",
        'years': [str(datetime.datetime.now().year)],  # Por defecto solo el año actual
        'currencies': [],  # Lista vacía para obtener todas las monedas
        'wait_timeout': 30,
        'credentials_path': "curso-sbk-bigdata-2025-935573af6e18.json",
        'spreadsheet_url': "https://docs.google.com/spreadsheets/d/1dIFZlNsDNP0vh1JLusy6wCRnDffeaPo3UFkQL1y9OtU/edit",
        'sheet_name': "Cotizaciones"
    }
    
    scraper = BCPScraper(config)
    try:
        start_time = time.time()
        raw_data = scraper.scrape_data()
        elapsed_time = time.time() - start_time
        logger.info(f"Scraping completado en {elapsed_time:.2f} segundos")
    finally:
        scraper.close()
    
    processed_data = DataProcessor.process_data(raw_data)
    
    if processed_data is None or processed_data.empty:
        logger.error("No hay datos válidos para exportar")
        return
    
    logger.info(f"\nMuestra de datos:\n{processed_data.head(10)}")
    logger.info(f"\nTotal registros nuevos: {len(processed_data)}")
    
    updater = GoogleSheetsUpdater(config['credentials_path'])
    if updater.update_sheet(
        processed_data, 
        config['spreadsheet_url'], 
        config['sheet_name'],
        config['years'] if config['years'] else None,  # Pasar None si se obtuvieron todos los años
        config['currencies'] if config['currencies'] else None  # Pasar None si se obtuvieron todas las monedas
    ):
        logger.info("Datos actualizados exitosamente")
    else:
        logger.error("Error en la actualización")


if __name__ == "__main__":
    main()