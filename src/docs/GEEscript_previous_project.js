/* JS EE from previous project Aaron Linder && Toby (Omdena) */

var cp = ee.FeatureCollection("users/omdenafarmhand/Chilli_plots"),
    ncp = ee.FeatureCollection("users/omdenafarmhand/Non-Chilli_plots"),
    geometry = 
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[76.69059978645427, 15.276378987596015],
          [76.69059978645427, 15.158110262066302],
          [76.79393993538005, 15.158110262066302],
          [76.79393993538005, 15.276378987596015]]], null, false),
    geometry2 = 
    /* color: #98ff00 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[76.69645971835375, 15.324951740258655],
          [76.69645971835375, 15.248781435412727],
          [76.75430960238695, 15.248781435412727],
          [76.75430960238695, 15.324951740258655]]], null, false);




var Start_period = ee.Date('2019-07-01');
var End_period = ee.Date('2020-03-15');



Map.setOptions('satellite');

// var geometry =cp.merge(ncp);
Map.centerObject(geometry,12);
Map.addLayer(geometry,{},"Geomtery");


// var s2Collection =  ee.ImageCollection("COPERNICUS/S2_SR").filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',10)).filter(ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2B')).filter(ee.Filter.eq('SENSING_ORBIT_DIRECTION', 'DESCENDING'))
//   .filterBounds(geometry)
//   .filterDate(Start_period,End_period);
  
var s2Collection =  ee.ImageCollection("COPERNICUS/S2_SR")
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',10))
  // .filter(ee.Filter.eq('SPACECRAFT_NAME', 'Sentinel-2B'))
  // .filter(ee.Filter.eq('SENSING_ORBIT_DIRECTION', 'DESCENDING'))
  .filterBounds(geometry)
  .filterDate(Start_period,End_period);
print('first S2 Collection :',s2Collection);
print('total number of images found :',s2Collection.size());
  
var maskcloud1 = function(image) {
var QA60 = image.select(['QA60']);
return image.updateMask(QA60.lt(1));
};

s2Collection = s2Collection.map(maskcloud1);

  

var fndvi = function(image){
  var ndvi = image.expression(
  "(NIR-RED)/(NIR+RED)",
  {
    RED: image.select('B4').multiply(0.0001),
    NIR : image.select('B5').multiply(0.0001)
    
  });// okay;
  var ndf = ndvi.rename('NDVI');
  var results = ndf.copyProperties(image, ['system:time_start']);
  return image.addBands(results);
};

var addEVI=function(image){
  var EVI = image.expression(
      '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
      {
      'NIR' : image.select('B8').divide(10000),
      'RED' : image.select('B4').divide(10000),
      'BLUE': image.select('B2').divide(10000)}).rename('EVI');
      return image.addBands(EVI);
};

var addSAVI = function(image){
  var SAVI = image.expression(
    '(B08 - B04) / (B08 + B04 + 0.428) * (1.0 + 0.428)',
    {
      'B08' : image.select('B8').divide(10000),
      'B04' : image.select('B4').divide(10000)}).rename('SAVI');
      return image.addBands(SAVI);
  
};

var addNDWI = function(image){
  var NDWI = image.expression(
    '(B03 - B08) / (B03 + B08)',
    {
      'B03' :image.select('B3').divide(10000),
      'B08' :image.select('B8').divide(10000)}).rename('NDWI');
      return image.addBands(NDWI);
};


// var addLabel = function(image){
//   var label = corn_2020.rename('Label');
//   return image.addBands(label);
// };



s2Collection=s2Collection.map(fndvi);
s2Collection =s2Collection.map(addEVI);
s2Collection= s2Collection.map(addSAVI);
s2Collection = s2Collection.map(addNDWI);
// s2Collection= s2Collection.map(addLabel);


print('Sentinel 2 Collection preprocessed:',s2Collection);

var mndvi = s2Collection.select('NDVI','EVI','SAVI','NDWI');
print('Monthly NDVI stack :',mndvi);


var opt_bands = mndvi.toList(ee.Number(mndvi.size()));
print("list of bands",opt_bands);


print("Optical Bands for Modelling",opt_bands.length().getInfo());




var training_image = ee.Image(opt_bands.get(0));


for (var i = 1; i < opt_bands.length().getInfo(); i++) {
  var myMap = ee.Image(opt_bands.get(i));
  training_image = training_image.addBands(myMap);
}


print('Training image optical',training_image);
Map.addLayer(training_image,{},'Training Image');
// Map.addLayer(training_image.clip(geometry),{},'Training image');
var training_image = training_image.toFloat()
Map.addLayer(cp,{color:'red'},'Chilli Plots');
Map.addLayer(ncp,{color :'green'},' Non Chilli Plots')



var ClassChart = ui.Chart.image.series({
  imageCollection: s2Collection.select('NDVI'),
  region: cp,
  reducer: ee.Reducer.median(),
  scale: 100,
})
  .setOptions({
      title: 'Summer Corn Not Detected NDVI value',
      hAxis: {'title': 'Date'},
      vAxis: {'title': 'Area of NDVI Value '},
      lineWidth: 2
    })

//Set the postion of the chart and add it to the map    
ClassChart.style().set({
    position: 'bottom-right',
    width: '500px',
    height: '300px'
  });
  
print(ClassChart);


var chart2 =
    ui.Chart.image
        .seriesByRegion({
          imageCollection:s2Collection.select('NDVI'),
          band: 'NDVI',
          regions: cp,
          reducer: ee.Reducer.median(),
          scale: 30,
          //seriesProperty: 'landcover',
          //xProperty: 'system:time_start'
        });
      
        
print(chart2);

// Export a cloud-optimized GeoTIFF.
Export.image.toDrive({
  image: training_image,
  description: 'time series image',
  region: geometry2,
  crs : 'EPSG:32643',
  fileFormat: 'GeoTIFF',
  scale: 10
});