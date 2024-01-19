"""
Generate a folium map of the data in the data directory to
visualize the results of the phenology model in a HTML file.

Author: Lukas Graf (lukas.graf@terensis.io)
"""

import branca.colormap as bcm
import folium
import geopandas as gpd
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import yaml

from branca.element import Element, MacroElement, Template
from eodal.core.band import Band, GeoInfo
from folium.raster_layers import ImageOverlay
from pathlib import Path
from rasterio.coords import BoundingBox


class BindColormap(MacroElement):
    """Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    """

    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
        self.layer = layer
        self.colormap = colormap
        self._template = Template(
            """
        {% macro script(this, kwargs) %}
            {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
            {{this._parent.get_name()}}.on('overlayadd', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
                }});
            {{this._parent.get_name()}}.on('overlayremove', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
                }});
        {% endmacro %}
        """
        )  # noqa


def get_textbox_css():
    return """
    {% macro html(this, kwargs) %}
    <!doctype html>
    <html lang="de">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Biomasse Demo Terensis</title>
        <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css" integrity="sha512-MV7K8+y+gLIBoVD59lQIYicR65iaqukzvf/nwasF0nqhPay5w/9lJmVM2hMDcnK1OnMGCdVK+iQrJ7lzPJQd1w==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
        <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
        <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

        <script>
        $( function() {
            $( "#textbox" ).draggable({
            start: function (event, ui) {
                $(this).css({
                right: "auto",
                top: "auto",
                bottom: "auto"
                });
            }
            });
        });
        </script>
    </head>

    <body>
        <div id="textbox" class="textbox">
        <div class="textbox-title">Getreide Biomasse und Ernteertrag</div>
        <div class="textbox-content">
            <pre>
            Die Karte zeigt unser Biomasse-Modell für ausgesuchte 
            Getreide-Felder während fünf Terminen kurz vor dem 
            Ährenschieben im Jahr 2021. Zudem ist der vom Modell geschätzte
            Ernteertrag dargestellt.
            Die zeitliche Auflösung des Modells ist 5 Tage, die räumliche
            Auflösung beträgt 10 Meter.
            </pre>
        </div>
        </div>
        <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 250px; 
                height: 80px; 
                z-index:9999; 
                font-size:14px;
                text-align: center;">
        <img src="https://github.com/terensis/ww_phenology_demo/raw/main/resources/terensis_logo.png" alt="Terensis" style="width: 250px; height: auto;">
    </div>
    </body>
    </html>

    <style type='text/css'>
    .textbox {
        position: absolute;
        z-index:9999;
        border-radius:4px;
        background: rgba( 90, 114, 71, 0.25 );
        box-shadow: 0 8px 32px 0 rgba( 90, 114, 71, 0.37 );
        backdrop-filter: blur( 4px );
        -webkit-backdrop-filter: blur( 4px );
        border: 4px solid rgba( 90, 114, 71, 0.2 );
        padding: 10px;
        font-size: 14px;
        right: 20px;
        bottom: 20px;
        color: #5a7247;
    }
    .textbox .textbox-title {
        color: black;
        text-align: center;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 22px;
        }
    .textbox .textbox-content {
        color: black;
        text-align: left;
        margin-bottom: 5px;
        font-size: 14px;
        }
    </style>
    {% endmacro %}
    """


def generate_folium_map(
    data_dir: Path,
    output_dir: Path,
    output_name: str = "index.html",
    vmin: float = 1.0,
    vmax: float = 13.0,
) -> None:
    """
    Generate a folium map of the data in the data directory to
    visualize the results of the growth model in a HTML file.

    :param data_dir: directory with the data (geojson files).
    :param output_dir: directory for writing outputs to.
    :param output_name: name of the output file.
    :param vmin: minimum value for the colormap (t/ha).
    :param vmax: maximum value for the colormap (t/ha).
    """
    # prepare the GLAI data
    field_calendar = gpd.read_file(
        data_dir.joinpath("2021/2021_Winterweizen-Triticale.gpkg")
    )

    fpath_growth_model_res = data_dir.joinpath("2021/2021_lai.npz")

    with np.load(fpath_growth_model_res) as src:
        growth_model_arr = src["interpolated_trait"]

    # load the GeoInfo yaml
    fpath_geoinfo_yaml = data_dir.joinpath("2021/2021_lai.yaml")
    with open(fpath_geoinfo_yaml, "r") as src:
        geo_info_dict = yaml.safe_load(src)

    # get the timestamps
    fpath_timestamps = data_dir.joinpath("2021/2021_lai.csv")
    timestamps = pd.read_csv(fpath_timestamps)
    timestamps.columns = ["date"]
    timestamps["date"] = pd.to_datetime(timestamps["date"])

    # select three dates during the season
    selected_indices = [30, 35, 40, 45, 50]

    # create map
    m = folium.Map(
        location=[46.98, 7.07],
        zoom_start=15,
        tiles="cartodbpositron",
        attr="© Terensis (2023). Basemap data © CartoDB",
    )

    # get the grain yield raster
    fpath_yield_map = data_dir.joinpath("2021/grain_yield_2021.tif")
    grain_yield = Band.from_rasterio(
        fpath_yield_map, band_idx=1)
    # get the mean grain yield per parcel
    grain_yield_stats = pd.DataFrame(grain_yield.reduce(by=field_calendar))
    field_calendar.loc[:, "grain_yield"] = \
        grain_yield_stats["mean"].values * 0.01
    field_calendar.loc[:, "grain_yield"] = \
        field_calendar["grain_yield"].round(2)
    field_calendar.loc[:, "crop_type"] = field_calendar.apply(
        lambda x: x.crop_type + f' (Mittlerer Ertrag: {x.grain_yield} t/ha)',
        axis=1
    )

    # display the field parcel boundaries
    field_calendar = field_calendar.to_crs(epsg=3857)
    # add to map (exclude timestamp column)
    field_calendar = field_calendar.drop(columns=["harvest_date"])
    folium.GeoJson(
        field_calendar,
        name="Feldgrenzen",
        style_function=lambda x: {
            "color": "black",
            "weight": 2,
            "fillOpacity": 0,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=["crop_type"],
            aliases=["Kultur"],
            localize=True,
        ),
    ).add_to(m)

    # Add custom Terensis style
    textbox_css = get_textbox_css()
    my_custom_style = MacroElement()
    my_custom_style._template = Template(textbox_css)
    # Adding to the map
    m.get_root().add_child(my_custom_style)

    # add data
    idx = 0
    for selected_index in selected_indices:
        # get the date
        date = timestamps["date"].iloc[selected_index].date()

        # plot the raster values using ImageOverlay
        img = growth_model_arr[selected_index, :, :]
        # convert to biomass
        img = 2.31 * np.exp(0.25 * img) - 1.61
        # reproject to WGS84
        geo_info = GeoInfo(**geo_info_dict)
        band = Band(values=img, geo_info=geo_info, band_name="Biomass", nodata=np.nan)
        img_repr = band.reproject(target_crs=4326, nodata_src=np.nan, nodata_dst=np.nan)
        bounds = BoundingBox(*img_repr.bounds.exterior.bounds)
        img_repr = img_repr.values
        # no-data handling. TODO: this should be fixed in EOdal
        img_repr[img_repr == 1.0483531] = np.nan
        img_repr[img_repr == 1.0321578] = np.nan
        img_repr[img_repr == 1.0402092] = np.nan
        img_repr[img_repr == 1.0521408] = np.nan
        img_repr[img_repr == 1.0451645] = np.nan
        img_repr = np.clip((img_repr - vmin) / (vmax - vmin), 0, 1)

        show = idx == 0
        bm = ImageOverlay(
            image=img_repr,
            name=f"{date}",
            opacity=1,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            interactive=True,
            cross_origin=False,
            zindex=1,
            colormap=cm.viridis,
            mercator_project=False,
            show=show,
        )

        m.add_child(bm)
        idx += 1

    # add grain yield map
    # reprojection to WGS84
    img_repr = grain_yield.reproject(
        target_crs=4326,
        nodata_src=grain_yield.nodata,
        nodata_dst=grain_yield.nodata)
    bounds = BoundingBox(*img_repr.bounds.exterior.bounds)
    img_repr = img_repr.values
    img_repr[img_repr == grain_yield.nodata] = np.nan
    img_repr *= 0.01  # convert to t/ha (from g/m2)
    # stretch to [0, 1]
    img_repr = np.clip((img_repr - vmin) / (vmax - vmin), 0, 1)
    bm = ImageOverlay(
        image=img_repr,
        name="Grain Yield",
        opacity=1,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        interactive=False,
        cross_origin=False,
        zindex=1,
        colormap=cm.viridis,
        mercator_project=False,
        show=False,
    )
    m.add_child(bm)

    # add colorbar
    colormap = bcm.LinearColormap(
        colors=[cm.viridis(x) for x in np.linspace(0, 1, num=256)],
        vmin=vmin,
        vmax=vmax,
        caption="Biomasse (t/ha)",
    )
    colormap.add_to(m)

    # save map
    m.add_child(folium.map.LayerControl(collapsed=False))
    m.save(output_dir.joinpath(output_name))


if __name__ == "__main__":
    data_dir = Path("data")
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)
    generate_folium_map(data_dir, output_dir)
