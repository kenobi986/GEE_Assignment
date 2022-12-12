// import shapefile
var shp = ee.FeatureCollection(Polygon)
//Map.addLayer(shp, {}, 'My Ploygon')
//Filter date range
var IMG = data3C.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20).filterDate('2017-03-01', '2017-12-30').filterBounds(shp);
//var image = data3C.filterDate('2017-03-28', '2017-06-30')
//var Bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','B10','QA60']
//var Bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','B10','QA60']
//var Bands = ['B2','B3','B4','B8','QA60']
//Select the bands
//var dataset = image.filterBounds(shp).select(['B2','B3','B4','B5']).first(dataset);
//print(dataset);
//Map.addLayer(dataset);

//Filter to get the least cloudy image for the area of interest
//var Lesscloudy = dataset.sort('CLOUD_COVERAGE_ASSESSMENT').first();
//print(Lesscloudy);
//Map.addLayer(Lesscloudy);

//var IMG = image.filterBounds(shp).select(Bands).sort('CLOUD_COVERAGE_ASSESSMENT');
//print(IMG);
//Map.addLayer(IMG);


// Function to keep only vegetation and soil pixels
//function SoilVeg(image) {
  // Select SCL layer
//  var scl = image.select('SCL'); 
  // Select vegetation and soil pixels
//  var veg = scl.eq(4); // 4 = Vegetation
//  var soil = scl.eq(5); // 5 = Bare soils
  // Mask if not veg or soil
//  var mask = (veg.neq(1)).or(soil.neq(1));
//  return image.updateMask(mask);
//}

// Apply custom filter to S2 collection
//var IMG2 = IMG.map(SoilVeg);


// Filter defined here: 
// https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#description

function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask);
}

// Function to compute NDVI and add result as new band
var addNDVI = function(image) {
return image.addBands(image.normalizedDifference(['B8', 'B4']));
};

// Add NDVI band to image collection
var IMG1 = IMG.map(addNDVI);
var IMG2 = IMG1.map(maskS2clouds);

var evoNDVI = ui.Chart.image.seriesByRegion(
  IMG2,                // Image collection
  shp,      // Region
  ee.Reducer.mean(), // Type of reducer to apply
  'nd',              // Band
  1);               // Scale
// Apply second filter

var plotNDVI = evoNDVI                    // Data
    .setChartType('LineChart')            // Type of plot
    .setOptions({                         // Plot customization
      interpolateNulls: true,
      lineWidth: 1,
      pointSize: 3,
      title: 'NDVI annual evolution',
      hAxis: {title: 'Date'},
      vAxis: {title: 'NDVI'}
});

print(plotNDVI)