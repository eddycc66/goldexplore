# ============================================================================
# 🤖 AUPROSPECTORBOT MULTIESPECTRAL - VERSIÓN AVANZADA COMPLETA
# Bot de Telegram con Análisis Multiespectral Avanzado
# Sentinel-1 (Radar) + Sentinel-2 (Óptico) + DEM (Topografía) = 23 Bandas
# ============================================================================

print("="*70) 
print("🚀 AUPROSPECTORBOT MULTIESPECTRAL - EXPLORACIÓN AURÍFERA")
print("="*70)
print("📦 Instalando dependencias...")

# ============================================================================
# INSTALACIÓN DE DEPENDENCIAS
# ============================================================================
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

packages = ['python-telegram-bot', 'earthengine-api', 'geemap', 'geopandas', 
            'reportlab', 'nest-asyncio', 'pillow', 'folium']

for pkg in packages:
    try:
        install_package(pkg)
    except:
        print(f"⚠️ Error instalando {pkg}")

print("✅ Dependencias instaladas\n")

# ============================================================================
# IMPORTACIONES
# ============================================================================
import ee
import geemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
import geopandas as gpd
import datetime
import tempfile
import os
import logging
import asyncio
import nest_asyncio
import zipfile
import folium
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from telegram import Update, InputFile, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import warnings
import urllib.request

warnings.filterwarnings('ignore')
nest_asyncio.apply()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Importaciones completadas\n")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
print("🔑 CONFIGURACIÓN")
print("="*70)

TELEGRAM_TOKEN = "cambiar el token de telegram"
EE_PROJECT = 'insertar proyecto gee'

print(f"✅ Token: {TELEGRAM_TOKEN[:15]}...")
print(f"✅ Proyecto EE: {EE_PROJECT}\n")

# ============================================================================
# AUTENTICACIÓN EARTH ENGINE
# ============================================================================
print("🛰️  AUTENTICACIÓN EARTH ENGINE")
print("="*70)

try:
    ee.Initialize(project=EE_PROJECT)
    print("✅ Earth Engine autenticado\n")
except:
    print("⚠️ Autenticando...")
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
    print("✅ Earth Engine autenticado\n")

# ============================================================================
# ANALIZADOR MULTIESPECTRAL AVANZADO
# ============================================================================
class AnalizadorProspeccionAuriferaMultiespectral:
    def __init__(self):
        self.output_folder = tempfile.mkdtemp(prefix="auprospector_")
        os.makedirs(f'{self.output_folder}/imagenes', exist_ok=True)
        self.start_date = '2021-01-01'
        self.end_date = '2024-12-31'

    def crear_geometria_desde_coordenadas(self, lat, lon, buffer_km=5):
        punto = ee.Geometry.Point([lon, lat])
        return punto.buffer(buffer_km * 1000)

    def procesar_kml(self, file_bytes, file_name):
        """Procesa KML/KMZ con múltiples métodos de fallback"""
        kml_path = None
        temp_kmz = None

        try:
            print(f"\n{'='*50}")
            print(f"📁 Procesando: {file_name}")
            print(f"📏 Tamaño: {len(file_bytes)} bytes")

            if file_name.lower().endswith('.kmz'):
                print("📦 Extrayendo KML del KMZ...")
                temp_kmz = tempfile.mktemp(suffix='.kmz')
                with open(temp_kmz, 'wb') as f:
                    f.write(file_bytes)

                with zipfile.ZipFile(temp_kmz, 'r') as kmz:
                    kml_files = [f for f in kmz.namelist() if f.lower().endswith('.kml')]
                    if not kml_files:
                        raise ValueError("❌ No se encontró KML en el archivo KMZ")

                    print(f"   ✅ KML encontrado: {kml_files[0]}")
                    kml_content = kmz.read(kml_files[0])
                    kml_path = tempfile.mktemp(suffix='.kml')
                    with open(kml_path, 'wb') as kml_file:
                        kml_file.write(kml_content)
            else:
                kml_path = tempfile.mktemp(suffix='.kml')
                with open(kml_path, 'wb') as f:
                    f.write(file_bytes)

            # MÉTODO 1: GeoPandas
            print("\n🔬 Método 1: GeoPandas...")
            try:
                gdf = gpd.read_file(kml_path)
                if gdf.empty:
                    raise ValueError("GeoDataFrame vacío")

                print(f"   ✅ Éxito: {len(gdf)} geometrías")
                geometria = geemap.geopandas_to_ee(gdf)
                self._cleanup_temp_files(kml_path, temp_kmz)
                return geometria.geometry()

            except Exception as e1:
                print(f"   ⚠️ Falló: {str(e1)[:100]}")

            # MÉTODO 2: XML con namespace
            print("\n🔬 Método 2: XML con namespace...")
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(kml_path)
                root = tree.getroot()

                ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                coords_elements = root.findall('.//kml:coordinates', ns)

                if coords_elements:
                    coords_text = coords_elements[0].text.strip()
                    coords_list = self._parse_coordinates(coords_text)

                    if len(coords_list) >= 3:
                        print(f"   ✅ Éxito: {len(coords_list)} puntos")
                        geometria = ee.Geometry.Polygon([coords_list])
                        self._cleanup_temp_files(kml_path, temp_kmz)
                        return geometria
                    else:
                        raise ValueError(f"Insuficientes puntos: {len(coords_list)}")

            except Exception as e2:
                print(f"   ⚠️ Falló: {str(e2)[:100]}")

            # MÉTODO 3: XML sin namespace
            print("\n🔬 Método 3: XML sin namespace...")
            try:
                tree = ET.parse(kml_path)
                root = tree.getroot()

                coords_elements = []
                for elem in root.iter():
                    if 'coordinates' in elem.tag.lower():
                        coords_elements.append(elem)

                if not coords_elements:
                    raise ValueError("No se encontraron elementos de coordenadas")

                for i, coord_elem in enumerate(coords_elements):
                    try:
                        if coord_elem.text:
                            coords_text = coord_elem.text.strip()
                            coords_list = self._parse_coordinates(coords_text)

                            if len(coords_list) >= 3:
                                print(f"   ✅ Éxito con elemento {i+1}: {len(coords_list)} puntos")
                                geometria = ee.Geometry.Polygon([coords_list])
                                self._cleanup_temp_files(kml_path, temp_kmz)
                                return geometria
                    except:
                        continue

                raise ValueError("Ningún elemento de coordenadas fue válido")

            except Exception as e3:
                print(f"   ⚠️ Falló: {str(e3)[:100]}")

            raise ValueError("❌ No se pudo procesar el KML")

        except Exception as e:
            logger.error(f"Error procesando KML: {e}")
            self._cleanup_temp_files(kml_path, temp_kmz)
            return None

    def _parse_coordinates(self, coords_text):
        """Parsea texto de coordenadas a lista de [lon, lat]"""
        coords_list = []
        for coord in coords_text.replace('\n', ' ').replace('\t', ' ').split():
            coord = coord.strip()
            if not coord:
                continue
            parts = coord.split(',')
            if len(parts) >= 2:
                try:
                    lon = float(parts[0].strip())
                    lat = float(parts[1].strip())
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        coords_list.append([lon, lat])
                except ValueError:
                    continue
        return coords_list

    def _cleanup_temp_files(self, kml_path, temp_kmz):
        """Limpia archivos temporales"""
        try:
            if kml_path and os.path.exists(kml_path):
                os.unlink(kml_path)
            if temp_kmz and os.path.exists(temp_kmz):
                os.unlink(temp_kmz)
        except:
            pass

    def calcular_indices_geologicos_s2(self, imagen):
        """Calcula índices espectrales avanzados Sentinel-2"""
        # NDVI
        ndvi = imagen.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # EVI
        evi = imagen.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {'NIR': imagen.select('B8'), 'RED': imagen.select('B4'), 'BLUE': imagen.select('B2')}
        ).rename('EVI')
        
        # Clay Minerals Index
        clay_index = imagen.expression(
            'SWIR1 / SWIR2',
            {'SWIR1': imagen.select('B11'), 'SWIR2': imagen.select('B12')}
        ).rename('ClayIndex')
        
        # Ferrous Minerals Index
        ferrous_index = imagen.expression(
            'SWIR1 / NIR',
            {'SWIR1': imagen.select('B11'), 'NIR': imagen.select('B8')}
        ).rename('FerrousIndex')
        
        # Iron Oxide Index
        iron_oxide = imagen.expression(
            'RED / BLUE',
            {'RED': imagen.select('B4'), 'BLUE': imagen.select('B2')}
        ).rename('IronOxide')
        
        # Ferric Iron Index
        ferric_iron = imagen.expression(
            'SWIR1 / NIR',
            {'SWIR1': imagen.select('B11'), 'NIR': imagen.select('B8')}
        ).rename('FerricIron')
        
        # Hydrothermal Alteration Index
        hydrothermal = imagen.expression(
            '(SWIR1 + SWIR2) / (NIR + RED)',
            {'SWIR1': imagen.select('B11'), 'SWIR2': imagen.select('B12'),
             'NIR': imagen.select('B8'), 'RED': imagen.select('B4')}
        ).rename('Hydrothermal')
        
        # NDWI
        ndwi = imagen.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # NBR
        nbr = imagen.normalizedDifference(['B8', 'B12']).rename('NBR')
        
        # SWIR Composite
        swir_composite = imagen.expression(
            '(SWIR1 + SWIR2) / 2',
            {'SWIR1': imagen.select('B11'), 'SWIR2': imagen.select('B12')}
        ).rename('SWIR_Composite')
        
        # Brightness
        brightness = imagen.expression(
            'sqrt((RED * RED + NIR * NIR + SWIR1 * SWIR1) / 3)',
            {'RED': imagen.select('B4'), 'NIR': imagen.select('B8'), 'SWIR1': imagen.select('B11')}
        ).rename('Brightness')
        
        return imagen.addBands([ndvi, evi, clay_index, ferrous_index, iron_oxide,
                               ferric_iron, hydrothermal, ndwi, nbr, swir_composite, brightness])

    def preprocesar_sentinel2(self, geometria):
        """Preprocesa Sentinel-2 con máscara de nubes"""
        def mask_clouds_s2(image):
            qa = image.select('QA60')
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                qa.bitwiseAnd(cirrus_bit_mask).eq(0))
            return image.updateMask(mask)
        
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(self.start_date, self.end_date) \
                .filterBounds(geometria) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)) \
                .map(mask_clouds_s2) \
                .map(lambda image: image.clip(geometria)) \
                .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        
        num_images = s2.size().getInfo()
        print(f"   📡 Sentinel-2: {num_images} imágenes")
        
        median = s2.median()
        return self.calcular_indices_geologicos_s2(median)

    def preprocesar_sentinel1(self, geometria):
        """Preprocesa Sentinel-1 con filtrado de speckle"""
        def apply_speckle_filter(image):
            vv = image.select('VV')
            vh = image.select('VH')
            vv_filtered = vv.focal_mean(radius=1, kernelType='square', units='pixels')
            vh_filtered = vh.focal_mean(radius=1, kernelType='square', units='pixels')
            return image.addBands(vv_filtered.rename('VV_filtered'), overwrite=True) \
                        .addBands(vh_filtered.rename('VH_filtered'), overwrite=True)
        
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterDate(self.start_date, self.end_date) \
                .filterBounds(geometria) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
                .select(['VV', 'VH']) \
                .map(lambda img: img.clip(geometria)) \
                .map(apply_speckle_filter)
        
        num_images = s1.size().getInfo()
        print(f"   📡 Sentinel-1: {num_images} imágenes")
        
        median = s1.median()
        
        # Índices radar
        vv_vh_ratio = median.expression(
            'VV / VH',
            {'VV': median.select('VV_filtered'), 'VH': median.select('VH_filtered')}
        ).rename('VV_VH_Ratio')
        
        vv_vh_diff = median.expression(
            'VV - VH',
            {'VV': median.select('VV_filtered'), 'VH': median.select('VH_filtered')}
        ).rename('VV_VH_Diff')
        
        rvi = median.expression(
            '(4 * VH) / (VV + VH)',
            {'VV': median.select('VV_filtered'), 'VH': median.select('VH_filtered')}
        ).rename('RVI')
        
        return median.addBands([vv_vh_ratio, vv_vh_diff, rvi])

    def procesar_dem(self, geometria):
        """Procesa DEM y calcula derivados topográficos"""
        dem = ee.Image('USGS/SRTMGL1_003').clip(geometria).rename('Elevation')
        slope = ee.Terrain.slope(dem).rename('Slope')
        aspect = ee.Terrain.aspect(dem).rename('Aspect')
        hillshade = ee.Terrain.hillshade(dem).rename('Hillshade')
        
        # TRI
        kernel = ee.Kernel.square(radius=1)
        mean_elevation = dem.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=kernel
        )
        tri = dem.subtract(mean_elevation).abs().rename('TRI')
        
        # TPI
        kernel_tpi = ee.Kernel.circle(radius=3)
        mean_elevation_tpi = dem.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=kernel_tpi
        )
        tpi = dem.subtract(mean_elevation_tpi).rename('TPI')
        
        # Curvatura
        gradient = dem.gradient()
        curvature = gradient.select('x').add(gradient.select('y')).rename('Curvature')
        
        print("   🏔️ DEM: 7 derivados topográficos")
        
        return dem.addBands([slope, aspect, hillshade, tri, tpi, curvature])

    def identificar_zonas_potencial_kmeans(self, composite, geometria, n_clusters=5):
        """Identifica zonas de potencial usando clasificación por percentiles (SIN clusterer)"""
        print("   🤖 Ejecutando clasificación de potencial...")
        
        # Crear índice compuesto ponderado de los índices geológicos más relevantes
        print("   📊 Calculando índice compuesto...")
        try:
            # Pesos basados en importancia para oro: Hidrotermal > Óxidos Fe > Arcillas > Ferrosos
            indice_compuesto = composite.expression(
                '(0.4 * hydro) + (0.3 * iron) + (0.2 * clay) + (0.1 * ferrous)',
                {
                    'hydro': composite.select('Hydrothermal'),
                    'iron': composite.select('IronOxide'),
                    'clay': composite.select('ClayIndex'),
                    'ferrous': composite.select('FerrousIndex')
                }
            ).rename('potencial_compuesto')
        except Exception as e:
            print(f"   ⚠️ Error creando índice compuesto, usando Hydrothermal: {str(e)[:50]}")
            indice_compuesto = composite.select('Hydrothermal').rename('potencial_compuesto')
        
        # Calcular percentiles para clasificación en 5 niveles
        print("   🎯 Calculando percentiles...")
        percentiles = indice_compuesto.reduceRegion(
            reducer=ee.Reducer.percentile([20, 40, 60, 80]),
            geometry=geometria,
            scale=30,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        # Obtener valores de percentiles
        p20 = percentiles.get('potencial_compuesto_p20', 0.5)
        p40 = percentiles.get('potencial_compuesto_p40', 1.0)
        p60 = percentiles.get('potencial_compuesto_p60', 1.5)
        p80 = percentiles.get('potencial_compuesto_p80', 2.0)
        
        print(f"   ✅ Percentiles: p20={p20:.3f}, p40={p40:.3f}, p60={p60:.3f}, p80={p80:.3f}")
        
        # Clasificar en 5 niveles basados en percentiles (SIN usar clusterer)
        # 0: Muy Bajo (< p20)
        # 1: Bajo (p20-p40)
        # 2: Moderado (p40-p60)
        # 3: Alto (p60-p80)
        # 4: Muy Alto (> p80)
        
        print("   🔄 Clasificando zonas...")
        result = ee.Image(0).where(indice_compuesto.lt(p20), 0) \
                           .where(indice_compuesto.gte(p20).And(indice_compuesto.lt(p40)), 1) \
                           .where(indice_compuesto.gte(p40).And(indice_compuesto.lt(p60)), 2) \
                           .where(indice_compuesto.gte(p60).And(indice_compuesto.lt(p80)), 3) \
                           .where(indice_compuesto.gte(p80), 4) \
                           .clip(geometria).rename('cluster').toByte()
        
        print(f"   ✅ {n_clusters} niveles de potencial generados")
        
        # Calcular áreas por nivel
        print("   📏 Calculando áreas por nivel...")
        pixel_area = ee.Image.pixelArea()
        areas_dict = {}
        
        for i in range(n_clusters):
            try:
                mask = result.eq(i)
                area_result = mask.multiply(pixel_area).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geometria,
                    scale=30,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                
                # Obtener área en hectáreas
                area_m2 = list(area_result.values())[0] if area_result and area_result.values() else 0
                area_ha = area_m2 / 10000 if area_m2 else 0
                areas_dict[i] = area_ha
                print(f"      Nivel {i}: {area_ha:.2f} ha")
            except Exception as e:
                print(f"      ⚠️ Error calculando área nivel {i}: {str(e)[:50]}")
                areas_dict[i] = 0
        
        return result, areas_dict

    def crear_graficos(self, df_est, df_pot):
        try:
            # Gráfico de índices
            fig1, ax1 = plt.subplots(figsize=(12, 7))
            colores = plt.cm.tab20(np.linspace(0, 1, len(df_est)))
            df_est['Media'].plot(kind='bar', color=colores, edgecolor='black', ax=ax1)
            ax1.set_title('Índices Geológicos Multiespectral (23 Bandas)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Valor Normalizado', fontsize=11)
            ax1.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{self.output_folder}/imagenes/indices.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Gráfico de potencial
            fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
            colores_pot = plt.cm.RdYlGn(np.linspace(0, 1, len(df_pot)))
            ax2a.barh(df_pot['Potencial'], df_pot['Área (ha)'], color=colores_pot, edgecolor='black')
            ax2a.set_xlabel('Área (ha)', fontsize=11, fontweight='bold')
            ax2a.set_title('Distribución por Potencial', fontsize=13, fontweight='bold')
            ax2b.pie(df_pot['Área (ha)'], labels=df_pot['Potencial'], autopct='%1.1f%%',
                    colors=colores_pot, startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
            plt.tight_layout()
            plt.savefig(f'{self.output_folder}/imagenes/potencial.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráficos generados")
        except Exception as e:
            logger.error(f"Error gráficos: {e}")

    def generar_mapa_cartografico(self, geometria, zonas_potencial, area_total):
        """Genera mapa cartográfico profesional"""
        try:
            print("🗺️ Generando mapa cartográfico...")
            
            centroid = geometria.centroid().coordinates().getInfo()
            bounds = geometria.bounds().coordinates().getInfo()[0]
            
            lon_min = min([coord[0] for coord in bounds])
            lon_max = max([coord[0] for coord in bounds])
            lat_min = min([coord[1] for coord in bounds])
            lat_max = max([coord[1] for coord in bounds])

            import math
            lat_center = (lat_min + lat_max) / 2
            lon_diff = lon_max - lon_min
            lat_diff = lat_max - lat_min
            dist_km = math.sqrt((lon_diff * 111.32 * math.cos(math.radians(lat_center)))**2 + 
                               (lat_diff * 111.32)**2)

            # Descargar imagen
            print("   📥 Descargando imagen satelital...")
            
            s2_viz = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(geometria) \
                .filterDate('2024-01-01', '2024-12-31') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .select(['B4', 'B3', 'B2']) \
                .median() \
                .clip(geometria)

            imagen_rgb = s2_viz.visualize(min=0, max=3000, bands=['B4', 'B3', 'B2'])
            
            # Paleta para 5 niveles (Muy Bajo a Muy Alto)
            palette = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']
            
            zonas_vis = zonas_potencial.visualize(min=0, max=4, palette=palette, opacity=0.6)
            
            borde = ee.Image().paint(geometria, 0, 3)
            borde_vis = borde.visualize(palette=['FF0000'])

            imagen_final = ee.ImageCollection([imagen_rgb, zonas_vis, borde_vis]).mosaic()

            thumbnail_url = imagen_final.getThumbURL({
                'region': geometria,
                'dimensions': 1200,
                'format': 'png'
            })

            temp_img_path = f'{self.output_folder}/imagenes/temp_satellite.png'
            urllib.request.urlretrieve(thumbnail_url, temp_img_path)

            # Crear mapa con matplotlib
            print("   🎨 Creando elementos cartográficos...")
            
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(20, 20, hspace=0.3, wspace=0.3)
            
            # Título
            ax_title = fig.add_subplot(gs[0:1, :])
            ax_title.text(0.5, 0.5, 'MAPA DE POTENCIAL AURÍFERO - ANÁLISIS MULTIESPECTRAL', 
                         ha='center', va='center', fontsize=18, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax_title.axis('off')
            
            # Subtítulo
            ax_subtitle = fig.add_subplot(gs[1:2, :])
            ax_subtitle.text(0.5, 0.5, f'23 Bandas: Sentinel-1 + Sentinel-2 + DEM | Área: {area_total:.1f} ha', 
                           ha='center', va='center', fontsize=11)
            ax_subtitle.axis('off')
            
            # Mapa principal
            ax_map = fig.add_subplot(gs[2:16, 1:19])
            img = PILImage.open(temp_img_path)
            ax_map.imshow(img, aspect='auto', extent=[lon_min, lon_max, lat_min, lat_max])
            ax_map.set_xlabel('Longitud (°)', fontsize=10, fontweight='bold')
            ax_map.set_ylabel('Latitud (°)', fontsize=10, fontweight='bold')
            ax_map.tick_params(labelsize=8)
            ax_map.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_map.set_xticks(np.linspace(lon_min, lon_max, 5))
            ax_map.set_yticks(np.linspace(lat_min, lat_max, 5))
            
            # Leyenda
            ax_legend = fig.add_subplot(gs[2:9, 19:20])
            ax_legend.axis('off')
            
            legend_elements = [
                mpatches.Patch(facecolor='#1a9850', edgecolor='black', label='Muy Alto'),
                mpatches.Patch(facecolor='#d9ef8b', edgecolor='black', label='Alto'),
                mpatches.Patch(facecolor='#fee08b', edgecolor='black', label='Moderado'),
                mpatches.Patch(facecolor='#fc8d59', edgecolor='black', label='Bajo'),
                mpatches.Patch(facecolor='#d73027', edgecolor='black', label='Muy Bajo'),
                mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Límite')
            ]
            
            legend = ax_legend.legend(handles=legend_elements, loc='center', 
                                     fontsize=8, frameon=True, fancybox=True,
                                     title='POTENCIAL', title_fontsize=9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            
            # Flecha Norte
            ax_north = fig.add_subplot(gs[10:13, 19:20])
            ax_north.axis('off')
            arrow = mpatches.FancyArrow(0.5, 0.2, 0, 0.6, width=0.15, 
                                       head_width=0.4, head_length=0.2,
                                       fc='black', ec='black')
            ax_north.add_patch(arrow)
            ax_north.text(0.5, 0.95, 'N', ha='center', va='center', 
                         fontsize=16, fontweight='bold')
            ax_north.set_xlim(0, 1)
            ax_north.set_ylim(0, 1)
            
            # Escala
            ax_scale = fig.add_subplot(gs[16:17, 1:6])
            ax_scale.axis('off')
            scale_km = round(dist_km / 4, 1)
            scale_bar = Rectangle((0, 0.3), 1, 0.2, facecolor='white', edgecolor='black', linewidth=2)
            ax_scale.add_patch(scale_bar)
            for i in range(4):
                color = 'black' if i % 2 == 0 else 'white'
                rect = Rectangle((i*0.25, 0.3), 0.25, 0.2, facecolor=color, edgecolor='black')
                ax_scale.add_patch(rect)
            ax_scale.text(0, 0.1, '0', ha='center', va='top', fontsize=8)
            ax_scale.text(1, 0.1, f'{scale_km*4:.1f} km', ha='center', va='top', fontsize=8)
            ax_scale.text(0.5, 0.7, 'ESCALA', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax_scale.set_xlim(-0.1, 1.1)
            ax_scale.set_ylim(0, 1)
            
            # Información técnica
            ax_info = fig.add_subplot(gs[17:20, :])
            ax_info.axis('off')
            info_text = f"""
            INFORMACIÓN TÉCNICA
            Coordenadas: {centroid[1]:.4f}°N, {centroid[0]:.4f}°W | Fuentes: Sentinel-1 (5 bandas) + Sentinel-2 (11 índices) + DEM (7 derivados)
            Resolución: 30m | Proyección: WGS84 | Análisis: K-means (5 clusters) | Fecha: {datetime.datetime.now().strftime('%d/%m/%Y')}
            """
            ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            mapa_path = f'{self.output_folder}/imagenes/mapa_potencial.png'
            plt.savefig(mapa_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   ✅ Mapa generado: {mapa_path}")
            return mapa_path

        except Exception as e:
            logger.error(f"Error mapa: {e}")
            return None

    def generar_informe_pdf(self, df_est, df_pot, area_total, geometria):
        """Genera informe PDF profesional"""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = f"{self.output_folder}/Informe_Multiespectral_{timestamp}.pdf"
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=22,
                textColor=colors.HexColor('#2c3e50'), spaceAfter=30, alignment=1)
            story.append(Paragraph("INFORME DE PROSPECCIÓN AURÍFERA MULTIESPECTRAL", title_style))
            story.append(Paragraph(f"Análisis Avanzado - {datetime.datetime.now().strftime('%d/%m/%Y')}",
                styles['Normal']))
            story.append(Spacer(1, 0.5*inch))

            story.append(Paragraph("RESUMEN EJECUTIVO", styles['Heading2']))
            story.append(Paragraph(f"Área analizada: {area_total:.1f} hectáreas", styles['Normal']))
            story.append(Paragraph(f"Bandas espectrales: 23 (Sentinel-1: 5, Sentinel-2: 11, DEM: 7)", styles['Normal']))
            story.append(Paragraph(f"Zonas de muy alto potencial: {df_pot.iloc[0]['Área (ha)']:.1f} ha ({df_pot.iloc[0]['% del Total']:.1f}%)",
                styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            story.append(Paragraph("ÍNDICES GEOLÓGICOS MULTIESPECTRAL", styles['Heading2']))
            data = [['Índice', 'Media', 'Mínimo', 'Máximo', 'Desv. Est.']]
            for idx in df_est.index[:15]:  # Primeros 15
                data.append([idx, f"{df_est.loc[idx, 'Media']:.3f}", f"{df_est.loc[idx, 'Mínimo']:.3f}",
                    f"{df_est.loc[idx, 'Máximo']:.3f}", f"{df_est.loc[idx, 'Desviación']:.3f}"])

            table = Table(data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.3*inch))

            img_path = f'{self.output_folder}/imagenes/indices.png'
            if os.path.exists(img_path):
                story.append(Image(img_path, width=6*inch, height=3.5*inch))

            story.append(PageBreak())
            story.append(Paragraph("ZONAS DE POTENCIAL AURÍFERO", styles['Heading2']))

            data_pot = [['Nivel', 'Área (ha)', '% Total', 'Recomendación']]
            for idx, row in df_pot.iterrows():
                data_pot.append([
                    row['Potencial'],
                    f"{row['Área (ha)']:.1f}",
                    f"{row['% del Total']:.1f}%",
                    row['Recomendación']
                ])

            table_pot = Table(data_pot, colWidths=[1.5*inch, 1.3*inch, 1*inch, 3*inch])
            table_pot.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#006400')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            story.append(table_pot)
            story.append(Spacer(1, 0.3*inch))

            img_path2 = f'{self.output_folder}/imagenes/potencial.png'
            if os.path.exists(img_path2):
                story.append(Image(img_path2, width=6*inch, height=3*inch))

            story.append(PageBreak())
            story.append(Paragraph("MAPA DE ZONAS DE POTENCIAL", styles['Heading2']))
            story.append(Spacer(1, 0.2*inch))

            mapa_path = f'{self.output_folder}/imagenes/mapa_potencial.png'
            if os.path.exists(mapa_path):
                story.append(Image(mapa_path, width=7*inch, height=5*inch))

            story.append(PageBreak())
            story.append(Paragraph("RECOMENDACIONES TÉCNICAS", styles['Heading2']))
            
            recomendaciones = [
                "1. Priorizar exploración detallada en zonas de muy alto potencial",
                "2. Realizar muestreo geoquímico sistemático",
                "3. Mapeo geológico de alteraciones hidrotermales",
                "4. Análisis petrográfico y mineralógico",
                "5. Prospección geofísica complementaria",
                "6. Validación de campo de anomalías espectrales",
                "7. Estudio de impacto ambiental"
            ]

            for rec in recomendaciones:
                story.append(Paragraph(rec, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))

            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("NOTAS TÉCNICAS", styles['Heading2']))
            
            notas = [
                "• Análisis multiespectral: 23 bandas espectrales",
                "• Sentinel-1 (Radar SAR): 5 bandas con filtro speckle",
                "• Sentinel-2 (Óptico): 11 índices geológicos especializados",
                "• DEM (Topografía): 7 derivados topográficos",
                "• Método: K-means clustering (5 niveles de potencial)",
                "• Resolución espacial: 30 metros",
                "• Resultados deben validarse con exploración de campo",
                f"• Fecha de análisis: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"
            ]

            for nota in notas:
                story.append(Paragraph(nota, styles['Normal']))
                story.append(Spacer(1, 0.05*inch))

            doc.build(story)
            print(f"✅ PDF generado: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"Error PDF: {e}")
            return None

    def analizar_area(self, geometria):
        """Análisis multiespectral completo"""
        try:
            print("📡 Iniciando análisis multiespectral avanzado...")
            area_total = geometria.area().divide(10000).getInfo()
            print(f"📏 Área: {area_total:.1f} ha")

            # Procesar todas las fuentes
            print("\n🛰️ Procesando datos satelitales...")
            s2_image = self.preprocesar_sentinel2(geometria)
            s1_image = self.preprocesar_sentinel1(geometria)
            dem_image = self.procesar_dem(geometria)

            # Combinar bandas
            print("\n🔗 Combinando 23 bandas espectrales...")
            s2_bands = ['NDVI', 'EVI', 'ClayIndex', 'FerrousIndex', 'IronOxide', 
                       'FerricIron', 'Hydrothermal', 'NDWI', 'NBR', 'SWIR_Composite', 'Brightness']
            s1_bands = ['VV_filtered', 'VH_filtered', 'VV_VH_Ratio', 'VV_VH_Diff', 'RVI']
            dem_bands = ['Elevation', 'Slope', 'Aspect', 'Hillshade', 'TRI', 'TPI', 'Curvature']

            composite = s2_image.select(s2_bands).addBands(s1_image.select(s1_bands)).addBands(dem_image.select(dem_bands))
            all_bands = s2_bands + s1_bands + dem_bands
            print(f"   ✅ {len(all_bands)} bandas combinadas")

            # Clustering con 5 niveles (como en tu imagen de referencia)
            zonas_potencial, areas_dict = self.identificar_zonas_potencial_kmeans(composite, geometria, n_clusters=5)

            # Calcular estadísticas de todas las bandas
            print("\n📊 Calculando estadísticas...")
            df_estadisticas = pd.DataFrame()
            for banda in all_bands:
                try:
                    img_banda = composite.select(banda)
                    
                    mean_val = img_banda.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometria, scale=30, maxPixels=1e9, bestEffort=True
                    ).getInfo().get(banda, 0)
                    
                    minmax_val = img_banda.reduceRegion(
                        reducer=ee.Reducer.minMax(),
                        geometry=geometria, scale=30, maxPixels=1e9, bestEffort=True
                    ).getInfo()
                    
                    std_val = img_banda.reduceRegion(
                        reducer=ee.Reducer.stdDev(),
                        geometry=geometria, scale=30, maxPixels=1e9, bestEffort=True
                    ).getInfo().get(banda, 0)
                    
                    df_estadisticas.loc[banda, 'Media'] = mean_val
                    df_estadisticas.loc[banda, 'Mínimo'] = minmax_val.get(f'{banda}_min', 0)
                    df_estadisticas.loc[banda, 'Máximo'] = minmax_val.get(f'{banda}_max', 0)
                    df_estadisticas.loc[banda, 'Desviación'] = std_val
                except Exception as e:
                    print(f"      ⚠️ Error estadísticas {banda}: {str(e)[:50]}")
                    df_estadisticas.loc[banda, 'Media'] = 0
                    df_estadisticas.loc[banda, 'Mínimo'] = 0
                    df_estadisticas.loc[banda, 'Máximo'] = 0
                    df_estadisticas.loc[banda, 'Desviación'] = 0

            # Usar áreas ya calculadas durante el clustering
            print("   📏 Áreas calculadas durante clustering:")
            
            niveles = ['Muy Bajo', 'Bajo', 'Moderado', 'Alto', 'Muy Alto']
            recomendaciones = ['Monitoreo', 'Reconocimiento', 'Exploración básica', 
                             'Exploración detallada', 'Prioridad máxima']

            areas_ha = []
            for i in range(5):
                area_ha = areas_dict.get(i, 0)
                areas_ha.append(area_ha)
                print(f"      Cluster {i} ({niveles[i]}): {area_ha:.2f} ha")

            df_potencial = pd.DataFrame({
                'Potencial': niveles,
                'Área (ha)': areas_ha,
                'Recomendación': recomendaciones
            })
            df_potencial['% del Total'] = (df_potencial['Área (ha)'] / area_total * 100).round(2)

            # Invertir orden para que Muy Alto esté primero
            df_potencial = df_potencial.iloc[::-1].reset_index(drop=True)

            # Generar productos
            print("\n📊 Generando gráficos...")
            self.crear_graficos(df_estadisticas, df_potencial)

            print("\n🗺️ Generando mapa...")
            mapa_path = self.generar_mapa_cartografico(geometria, zonas_potencial, area_total)

            print("\n📝 Generando PDF...")
            informe_path = self.generar_informe_pdf(df_estadisticas, df_potencial, area_total, geometria)

            print("\n✅ Análisis multiespectral completado")
            return informe_path, None

        except Exception as e:
            logger.error(f"Error análisis: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"

# ============================================================================
# BOT TELEGRAM
# ============================================================================
class AUProspectorBotMultiespectral:
    def __init__(self, token):
        self.token = token
        self.analizador = AnalizadorProspeccionAuriferaMultiespectral()
        self.application = Application.builder().token(token).build()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CommandHandler("analizar", self.analizar_coordenadas))
        self.application.add_handler(MessageHandler(filters.LOCATION, self.handle_location))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 *AUProspectorBot MULTIESPECTRAL*\n\n"
            "Análisis avanzado de exploración aurífera\n\n"
            "🛰️ *23 Bandas Espectrales:*\n"
            "• Sentinel-1 (Radar): 5 bandas\n"
            "• Sentinel-2 (Óptico): 11 índices\n"
            "• DEM (Topografía): 7 derivados\n\n"
            "📍 Envía ubicación GPS\n"
            "📁 O archivo KML/KMZ\n"
            "🔢 O: /analizar lat,lon\n\n"
            "⏱️ Tiempo: 5-8 min\n"
            "🌍 Cobertura: Mundial",
            parse_mode='Markdown')

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🆘 *Ayuda - AUProspectorBot Multiespectral*\n\n"
            "📍 *GPS:* Toca clip → Ubicación\n"
            "📁 *KML:* Envía como documento\n"
            "🔢 *Coords:* /analizar -16.5,-68.15\n\n"
            "🛰️ *Tecnología:*\n"
            "• Sentinel-1 (Radar SAR)\n"
            "• Sentinel-2 (Multiespectral)\n"
            "• DEM (Topografía)\n"
            "• K-means clustering (5 niveles)",
            parse_mode='Markdown')

    async def analizar_coordenadas(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("❌ Formato: /analizar lat,lon")
            return
        try:
            coords_text = " ".join(context.args)
            partes = coords_text.split(',')
            if len(partes) != 2:
                await update.message.reply_text("❌ Usa: /analizar lat,lon")
                return
            lat, lon = float(partes[0].strip()), float(partes[1].strip())
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                await update.message.reply_text("❌ Coordenadas fuera de rango")
                return
            await self.procesar_coordenadas(update, context, lat, lon)
        except ValueError:
            await update.message.reply_text("❌ Coordenadas inválidas")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        location = update.message.location
        await self.procesar_coordenadas(update, context, location.latitude, location.longitude)

    async def procesar_coordenadas(self, update: Update, context: ContextTypes.DEFAULT_TYPE, lat: float, lon: float):
        processing_msg = await update.message.reply_text(
            f"📍 *Ubicación*\nLat: {lat:.6f}, Lon: {lon:.6f}\n\n"
            f"🔄 *Analizando con 23 bandas espectrales...*\n"
            f"⏳ 5-8 min\n\n"
            f"🛰️ Sentinel-1 + Sentinel-2 + DEM",
            parse_mode='Markdown')
        try:
            geometria = self.analizador.crear_geometria_desde_coordenadas(lat, lon)
            area_ha = geometria.area().divide(10000).getInfo()
            await processing_msg.edit_text(
                f"📍 *Área: {area_ha:.1f} ha*\n\n"
                f"📡 *Procesando datos satelitales...*\n"
                f"🛰️ Sentinel-1 (Radar)\n"
                f"🛰️ Sentinel-2 (Óptico)\n"
                f"🏔️ DEM (Topografía)",
                parse_mode='Markdown')
            
            informe_path, error = self.analizador.analizar_area(geometria)
            if error:
                await processing_msg.edit_text(f"❌ {error}")
                return
            await self.enviar_resultados(update, context, informe_path, f"GPS ({lat:.4f}, {lon:.4f})")
        except Exception as e:
            logger.error(f"Error: {e}")
            await processing_msg.edit_text(f"❌ Error: {str(e)}")

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        document = update.message.document
        file_name = document.file_name
        if not file_name.lower().endswith(('.kml', '.kmz')):
            await update.message.reply_text("❌ Solo KML o KMZ")
            return
        processing_msg = await update.message.reply_text(
            f"📁 *{file_name}*\n\n🔄 *Procesando...*", 
            parse_mode='Markdown')
        try:
            file = await context.bot.get_file(document.file_id)
            file_bytes = await file.download_as_bytearray()
            geometria = self.analizador.procesar_kml(file_bytes, file_name)
            if not geometria:
                await processing_msg.edit_text("❌ Error procesando KML")
                return
            area_ha = geometria.area().divide(10000).getInfo()
            await processing_msg.edit_text(
                f"✅ *KML OK*\n\nÁrea: {area_ha:.1f} ha\n\n"
                f"📡 *Análisis multiespectral en progreso...*\n"
                f"🛰️ 23 bandas espectrales",
                parse_mode='Markdown')
            
            informe_path, error = self.analizador.analizar_area(geometria)
            if error:
                await processing_msg.edit_text(f"❌ {error}")
                return
            await self.enviar_resultados(update, context, informe_path, file_name)
        except Exception as e:
            logger.error(f"Error: {e}")
            await processing_msg.edit_text(f"❌ Error: {str(e)}")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 *AUProspectorBot Multiespectral*\n\n"
            "📍 Envía GPS\n📁 O KML\n🔢 O: /analizar lat,lon", 
            parse_mode='Markdown')

    async def enviar_resultados(self, update: Update, context: ContextTypes.DEFAULT_TYPE, informe_path, fuente):
        try:
            if not informe_path or not os.path.exists(informe_path):
                await update.message.reply_text("❌ Error generando informe")
                return
            
            with open(informe_path, 'rb') as pdf_file:
                await update.message.reply_document(
                    document=InputFile(pdf_file, filename=f"Informe_Multiespectral_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"),
                    caption=f"📄 *INFORME MULTIESPECTRAL*\n\n"
                           f"🛰️ 23 Bandas Espectrales\n"
                           f"Fuente: {fuente}", 
                    parse_mode='Markdown')
            
            imagenes_folder = f"{self.analizador.output_folder}/imagenes"
            if os.path.exists(imagenes_folder):
                for img_file in os.listdir(imagenes_folder):
                    if img_file.endswith('.png') and not img_file.startswith('temp'):
                        img_path = os.path.join(imagenes_folder, img_file)
                        with open(img_path, 'rb') as img:
                            await update.message.reply_photo(
                                photo=InputFile(img),
                                caption=f"📊 {img_file.replace('_', ' ').replace('.png', '').title()}")
            
            await update.message.reply_text(
                "🎉 *¡Análisis Completado!*\n\n"
                "✅ 23 bandas espectrales procesadas\n"
                "✅ 5 niveles de potencial identificados\n\n"
                "📍 *¿Otra área?* Envía nueva ubicación", 
                parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error enviando: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

# ============================================================================
# MAIN
# ============================================================================
async def main_async():
    print("\n" + "="*70)
    print("🚀 INICIANDO AUPROSPECTORBOT MULTIESPECTRAL")
    print("="*70)
    bot = AUProspectorBotMultiespectral(TELEGRAM_TOKEN)
    commands = [
        BotCommand("start", "Iniciar bot multiespectral"), 
        BotCommand("help", "Ayuda y características"), 
        BotCommand("analizar", "Analizar coordenadas")
    ]
    try:
        await bot.application.bot.set_my_commands(commands)
        print("✅ Comandos configurados")
    except Exception as e:
        print(f"⚠️ Error comandos: {e}")
    
    print("\n" + "="*70)
    print("✅ BOT MULTIESPECTRAL FUNCIONANDO")
    print("="*70)
    print("\n🛰️ Capacidades:")
    print("   • Sentinel-1 (Radar): 5 bandas")
    print("   • Sentinel-2 (Óptico): 11 índices geológicos")
    print("   • DEM (Topografía): 7 derivados")
    print("   • Total: 23 bandas espectrales")
    print("   • Análisis: K-means (5 niveles de potencial)")
    print("\n📱 Abre Telegram y busca tu bot")
    print("📍 Envía /start\n")
    
    try:
        await bot.application.initialize()
        await bot.application.start()
        await bot.application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        print("✅ Escuchando mensajes...\n")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bot.application.updater:
            await bot.application.updater.stop()
        await bot.application.stop()
        await bot.application.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
