import os
import pickle
import numpy
from osgeo import gdal, osr
from PIL import Image
from src.config import CACHE_PATH, METADATA_PATH, GEO_DATA_DIR
from src.single_layer_network import load_model, list_findings_2
from src.training_data import load_all_training_tiles, read_naip, NAIP_PIXEL_BUFFER
# from src.training_visualization import render_results_for_analysis

tf_preds_dir = os.path.join(GEO_DATA_DIR, 'tf_preds')


def rotate_and_flip(im):
    return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)


def main():
    """Generate image tiles corresponding to the predictions."""
    with open(CACHE_PATH + 'raster_data_paths.pickle', 'r') as infile:
        raster_data_paths = pickle.load(infile)

    with open(CACHE_PATH + METADATA_PATH, 'r') as infile:
        training_info = pickle.load(infile)

    with open(CACHE_PATH + 'model_metadata.pickle', 'r') as infile:
        model_info = pickle.load(infile)

    model = load_model(model_info['neural_net_type'], model_info['tile_size'],
                       len(model_info['bands']))
    bands = training_info['bands']

    for path in raster_data_paths:
        labels, images = load_all_training_tiles(path, bands)

        path_parts = path.split('/')
        filename = path_parts[len(path_parts) - 1]

        wp_preds = list_findings_2(labels, images, model)
        start_xs = [label[1] for label in labels]
        start_ys = [label[2] for label in labels]
        x_values = numpy.unique(start_xs).size
        y_values = numpy.unique(start_ys).size
        img = numpy.reshape(wp_preds, (x_values, y_values))  # wrong orientation!

        im_uint8 = Image.fromarray(img * 255).convert('RGB')
        png_file = os.path.splitext(filename)[0] + "_tf_pred.png"
        rotate_and_flip(im_uint8).save(os.path.join(tf_preds_dir, png_file))

        raster_dataset, bands_data = read_naip(path, bands)  # expensive!
        raster_prj = raster_dataset.GetProjection()
        epsg = osr.SpatialReference(wkt=raster_prj).GetAttrValue('AUTHORITY', 1)
        drv = gdal.GetDriverByName("GTiff")
        geotiff_file = os.path.splitext(filename)[0] + "_tf_pred.tif"
        ds = drv.Create(os.path.join(tf_preds_dir, geotiff_file),
                        x_values, y_values, 1, gdal.GDT_Float32)
        geo_transform = raster_dataset.GetGeoTransform()
        ds.SetGeoTransform((geo_transform[0] + NAIP_PIXEL_BUFFER, 64.0, 0.0,
                            geo_transform[3] - NAIP_PIXEL_BUFFER, 0.0, -64.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        ds.SetProjection(srs.ExportToWkt())
        im_flt32 = Image.fromarray(img)
        ds.GetRasterBand(1).WriteArray(numpy.asarray(rotate_and_flip(im_flt32)))
        ds.FlushCache()
        del ds

        # render_results_for_analysis([path], false_positives, fp_images, training_info['bands'],
        #                             training_info['tile_size'])


if __name__ == "__main__":
    main()
