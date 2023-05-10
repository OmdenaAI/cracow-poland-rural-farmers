


# Improving Digital Advisory Services for Rural Farmers using Predictive Analytics and Satellite Imagery

# The problem

We have seen traction in demand for rural digital advisory services, however current systems for digital advisory are focused on the broad delivery of extension services based on a large number of farmers. AI can revolutionize extension services through the provision of individualized advisory based on several data elements (on-farm data, satellite imagery, remote sensing, and GIS) thereby increasing the value for extension services to the individual farmer. Although use cases are being built in other development agencies and countries, we have not seen greater traction on AI and other technologies integration in IFAD-supported projects. This could be an opportunity to develop a Proof-of-Concept (POC) and develop a potential use case for scale.
<img width="710" alt="Screenshot 2023-05-04 at 1 13 52 PM" src="https://github.com/OmdenaAI/cracow-poland-rural-farmers/assets/9891291/945325aa-b714-4171-9573-28e94758c34d">

# We had no data!!!!

#How we created a GIS dataset with no data


Omdena Poland Chapter

#Introduction

Satellite imagery has become a valuable tool for monitoring crop health and identifying crop types. Copernicus, the European Union's Earth Observation Programme, provides free and open access to satellite data, making it an ideal source for creating a dataset for training an AI model to provide information on crop type and health for farmers in Poland. Creating a dataset for training an AI model involves collecting and labeling data. In this case, we will collect satellite images of agricultural fields in Poland and label them with information on crop type from the LUCAS Land Use and Coverage frame Survey.
Contrary to most GIS datasets, which provide raster at a regional or large area level, when it comes to identifying land usage, we understand that the details matter. That's why we work at the field level when it comes to identifying crop types. We recognize that each field is unique, and the crops grown on it can vary based on factors like soil quality, sun exposure, and moisture levels. By delimiting each field and labeling which type of crops are on it, we can monitor how it evolves over time and provide farmers with accurate and detailed information that they can use to make informed decisions about crop management. This micro-level approach allows us to create a customized solution for each field, ensuring that farmers get the most out of their crop data.

#Process and Workflow
<img width="1022" alt="Screenshot 2023-05-10 at 7 33 33 AM" src="https://github.com/OmdenaAI/cracow-poland-rural-farmers/assets/9891291/ec0d58a1-0d91-45cf-af85-616cad4fd9f3">

We first began working on exploratory data analysis, deciphering which satellite system is usable for downloading images as there were several choices: Landsat and ESA (European Space Agency) Satellites. We discovered that the Sentinel-2 Satellite was optimal with its NIR (Near InfraRed) sensor and was most suitable for crop contiguous or delineation detection. Second, we began image retrieval. We borrowed heavily from a master’s thesis called EveryField, in which we were able to find some techniques for downloading images and looking at the bands. There were some problems with delineation of fields using EveryField's techniques. One of the team members saw a new release of Meta code for Pytorch called “Segment Anything”,” which we began adapting for use in field delineation. After getting the mask, logic checks are made to prevent overlapping and faulty polygons to get good separation between polygons and no overlapping.
The purpose of field delineation is to train an LSTM (Long Short Term Memory) machine learning model to classify the crop type on a given polygon delimiting a crop field. 
#This is the phase we are at currently in the project:

    • Identify the area of interest: Find a 10km square subset of land where we have the most LUCAS survey points of crops.
    • Determine the survey points centroid: Calculate the centroid of the LUCAS survey points within the area of interest.
    •  Download satellite images: Use the 10km square area of interest to download Copernicus satellite images for the entire growing season.
    •  Select a smaller subset: From the downloaded satellite images, select a subset that is a 5km x 5km square centered at the centroid calculated in previous step.
    • Field segmentation: Use the RGB image of the subset as input for the "Segment-Anything" algorithm from Meta to segment the fields, cities, rivers, lakes, etc. Select the segmented area with the highest stability score corresponding to the LUCAS survey points and create a shapefile from it.
    • Feature engineering: Compute the Normalized Difference Water Index (NDWI) and Normalized Difference Vegetation Index (NDVI) scores for each pixel's average per month. Once the field area given by the shapefile is known, compute the average value of each score per crop field for each month and assign the crop type (i.e. Sugar Beet, wheat, etc.) from the LUCAS survey dataset for each field.
    • OUPUT: data frame ready for LSTM with each row corresponding to a survey field with features such as average NDVI etc and target crop type labels like wheat, corn, rye, etc.

So, this is the output of what we have been working on. It's just a DEMO with only one subset because we need a batch processing pipeline demonstration. This program would require terabytes of data to feed a model.
In the illustration below, on the left, the RGB subset image is presented, and on the right, the corresponding segmentation using META’s Pytorch is shown. As you can see, we have designated fields in our LUCAS dataset represented by red dots to which we have assigned labels. The top left red dot represents an unknown cereal, the bottom left is common wheat, and the right one is sugar beets. Fields on the South side of Poland are fairly small, and the META’s Pytorch did a good job of segmenting the field on a 5km x 5km subset. However, as soon as the subset is larger (for example, 10km x 10km), the segmentation becomes sub-optimal for our purposes!


<img width="1403" alt="Screenshot 2023-05-10 at 6 59 29 AM" src="https://github.com/OmdenaAI/cracow-poland-rural-farmers/assets/9891291/e7585e21-92e1-4985-afff-d75f56f7761a">

#The following illustration shows the standard deviation of the NDVI score averaged per month for the growing season:




#This is what is happening with the crop cycles when considering the average mean NDVI score:
<img width="804" alt="Screenshot 2023-05-10 at 6 59 53 AM" src="https://github.com/OmdenaAI/cracow-poland-rural-farmers/assets/9891291/cc0740ff-1591-4849-b0d7-23e11c04cffb">


As you can see, Poland plants "Winter Wheat" having two harvest seasons a year (i.e. Summer Wheat in the spring cycle, winter wheat in the fall cycle). Whereas beets or other plants are once-a-year crops and have their harvest in late September or early October. Using these signatures in an LSTM (Long Short-Term Memory), we can create a signature analysis of various European crops and determine other factors.
Unfortunately, we had limited access to resources. Being a volunteer project only, we would need a virtual machine that costs $150 a month and maybe 3TB of cloud storage, but we are not sure how much that would cost.

It is arguable that we may need more system resources as each satellite image is about 110km x 110km tiles of 1.2 GB, and we need to calculate how many 5km x 5km subsets we need to train the LSTM, which is beyond our cloud space capabilities. We are not currently sure of how much storage space we will need to be able to train a good LSTM, but it is certainly beyond 100GB and would require upgrades on our cloud accounts. Being an unfunded project (seeking funding, feel free to contact us), this would be a requirement at this point.

#Benefits
The benefits of using this protocol developed for field delineation and masking is that later on, the selected field mask can be used to calculate surface area and can be tied in with other LSTM for possible fertilizer calculations and also can show how well a field is doing. Also, yield can be calculated based on experimentation with other datasets and NDVI (Natural Differential Vegetative indexes) and other feature-rich indexes.
Additionally, there are expensive paid APIs ResUNet SentinelHub (300 Euro) that can do similar things that our software does, but our software is free and open-source. It is a good stepping stone for later projects.
Conclusion
We demonstrate the feasibility to identify and monitor fields at a micro-level, even for small field sizes. The notebooks will have to be adapted later to download more tiles and subsets, whatever calculation is sufficient enough to train the model. We are struggling to find a dataset for Poland in particular for field type and yield data with the granularity we were looking for.

#Future Work

What needs to be done at this point is the following:

    1. Download enough contiguous tiles matching the LUCAS dataset coordinates (we have narrowed down two areas, notebook will be included).
    2. Train the LSTM with the improved masks (needs a cloud storage upgrade).
    3. Select the fields with the LSTM and train the LSTM to accept all the crop codes with decent accuracy.
    4. Once the crop-bearing fields are selected, determine yields using extrapolation from other datasets, as well as calculating surface area for selected polygons and determining fertilizer use.
    5. Determine how much cost of seed is required by the surface area of the selected polygons, as farmers often operate on short-term loans to buy their seed in which they have to pay back after the harvest.

DataSets of Interest
There are some other datasets which give us similar information without the granularity
Raster data for several years in Europe from the CORINE Land Cover: https://land.copernicus.eu/pan-european/corine-land-cover/clc2018?tab=mapview
Pixel data mask GLOBAL LAND COVER in Europe for several years: https://lcviewer.vito.be/2018/Poland
Crop yield by provinces in Poland can be found in this portal: https://geo.stat.gov.pl/app/mapa/gus/c633a248-a094-0007-c39a-25d57ecdf0e4/?locale=en&mapview=52.108841%2C17.885166%2C7.28z#/
LUCAS dataset containing land usage survey information:
https://esdac.jrc.ec.europa.eu/projects/lucas
