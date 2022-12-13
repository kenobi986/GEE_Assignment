//Define Imports
var data3C = ee.ImageCollection("COPERNICUS/S2_HARMONIZED"),
    Polygon = ee.FeatureCollection("users/gorthisrikanth123/Polygon"),
    data2A = ee.ImageCollection("COPERNICUS/S2_SR"),
    data1C = ee.ImageCollection("COPERNICUS/S2_HARMONIZED");


//Import the dataset Shapefile and Sentinel-2 images 

var shp = ee.FeatureCollection(Polygon) //ShapeFile Feature Collection

//Filter out the cloud cover and boundary defined by shape file
var IMG = data1C.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20).filterDate('2017-01-01', '2017-12-30').filterBounds(shp);

print(shp)


//Mask the cloud in the images (Stadard Function)
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
   var date = image.date();
return image.addBands(image.normalizedDifference(['B8', 'B4']));
};

// Add NDVI band to image collection and remove clouds 
var IMG1 = IMG.map(maskS2clouds);
var IMG2 = IMG1.map(addNDVI);


//print the total number of images in the given year 2017
print(IMG2.count())


//Use reduce mean on the image masked by the shape file 
//set the date as the property for the resulting feature collection
var reduced = shp.map(function(featureCollection) {
  return IMG2.map(function(image) {
    //Calculate the reduce mean of the images over the shape file geometry
    var mean = image.reduceRegion({
      geometry: featureCollection.geometry(),
      reducer: ee.Reducer.mean(),
    });
    //Set the date as the property and return the feature collection
    return featureCollection.setMulti(mean).set({date: ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')})
  })
})

//Flatten the feature collection into table for export 
var table = reduced.flatten();

print(reduced.limit(100))

print(table.limit(100))

//Define the properties of interest
var properties = ee.List(['date','ID', 'nd', 'CropGrp', 'CropTyp']);
var desc = 'NDVI Mean';

//Export table to Asset 
Export.table.toAsset({
  collection: table.select(properties).filter(ee.Filter.and(
    ee.Filter.neq('ID', null),
    ee.Filter.neq('nd', null),
    ee.Filter.neq('CropGrp', null), 
    ee.Filter.neq('CropTyp', null),
    ee.Filter.neq('date', null))),
  description: desc, 
  assetId: desc
});