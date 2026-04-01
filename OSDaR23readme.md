# OSDaR23  |  Sequence  *'1_calibration_1.1'*

![color camera preview image for sequence group 01_calibration](readme_img/01_calibration_HF.png) ![lidar preview image for sequence group 01_calibration](readme_img/01_calibration_Lidar.png)
*Source: own illustration based on image / data by DB Netz AG*


## Project Info
![DZSF](readme_img/logo_dzsf.png) ![Digitale Schiene Deutschland](readme_img/logo_dsd.png) ![FusionSystems](readme_img/logo_fusionsystems.png)

The "Open Sensor Data for Rail 2023" (OSDaR23, [10.57806/9mv146r0](https://doi.org/10.57806/9mv146r0)) has been created in a joint research project by the [German Centre for Rail Traffic Research at the Federal Railway Authority (DZSF)](https://www.dzsf.bund.de), [Digitale Schiene Deutschland / DB Netz AG](http://digitale-schiene-deutschland.de), and [FusionSystems GmbH](https://www.fusionsystems.de). The [Research report (10.48755/dzsf.230012.01)](https://doi.org/10.48755/dzsf.230012.01) and the [Labeling Guide (10.48755/dzsf.230012.05)](https://doi.org/10.48755/dzsf.230012.05) can be obtained from the [DZSF website](https://www.dzsf.bund.de/SharedDocs/Standardartikel/DZSF/Projekte/Projekt_70_Reale_Datensaetze.html).

The data set consists of 45 sequences of annotated multi-sensor data (color camera, infrared camera, lidar, radar, localization, IMU). Data have been collected on different railway tracks in Hamburg, Germany.


## License Info
The "Open Sensor Data for Rail 2023" (OSDaR23, [10.57806/9mv146r0](https://doi.org/10.57806/9mv146r0)) is published by the [German Centre for Rail Traffic Research at the Federal Railway Authority (DZSF)](https://www.dzsf.bund.de). Annotation data (file type `.json`) are published under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode). Sensor data (file types `.png`, `.pcd`, and `.csv`) are published under [CC BY-SA 3.0 de](https://creativecommons.org/licenses/by-sa/3.0/de/legalcode).


## Further Info
The data set can be used in Python with the [RailLabel](https://github.com/DSD-DBS/raillabel) package published by DB Netz AG. 

The data set can be viewed, for example, with the [WebLabel Player](https://github.com/Vicomtech/weblabel) published by [Vicomtech Research Foundation](https://www.vicomtech.org/en/). *(Disclaimer: Vicomtech was not part of the research project and there are currently no further relationships between DZSF and Vicomtech.)*

![Vicomtech](readme_img/logo_vicomtech.png)



## Sequence Info for *'1_calibration_1.1'*

Dataset Version: **1.1.0**

### Description for Sequence Group *'1_calibration'*
The ego train runs onto a siding consisting of three tracks with buffer stops at the end. In subsequence 1.1, the ego train is on the right track. In 1.2 it runs on the left track. The distance to the buffer stops is the same. Common object classes are person (worker), road vehicle, catenary pole and track.

### Statistics for Sequence *'1_calibration_1.1'*
Number of Multisensor-Frames: `10`

Total file size: `946 MB` (compressed archive is 10-15% smaller)

Statistic of annotation objects:

    object class        number of annotations
    ------------------  ---------------------
    person                                166
    crowd                                   0
    train                                   0
    wagons                                  0
    bicycle                                20
    group of bicycles                      20
    motorcycle                              0
    road vehicle                          320
    animal                                  0
    group of animals                        0
    wheelchair                              0
    drag shoe                              79
    track                                 160
    transition                              0
    switch                                 32
    catenary pole                         540
    signal pole                            60
    signal                                200
    signal bridge                           0
    buffer stop                           165
    flame                                   0
    smoke                                   0
    ------------------  ---------------------
    total                               1 762

### Data Structure
Folders and files within the sequence are structured as follows.

    1_calibration_1.1/
        ├── ir_center/                                        images from the center infrared camera
        │    ├── [n  ]_[timestamp].png                        file name consists of counter + timestamp
        │    ├── [n+1]_[timestamp].png                          counters do not always start at '0', because
        │    ├── [n+2]_[timestamp].png                          sequences have been taken from longer sequences
        │    └── ...
        │
        ├── ir_left/                                          images from the left infrared camera
        ├── ir_right/                                         images from the right infrared camera
        ├── lidar/                                            merged point clouds from all lidar sensors
        │    ├── [n  ]_[timestamp].pcd                        file names correspond to data from other sensors
        │    ├── [n+1]_[timestamp].pcd
        │    └── ...
        │
        ├── novatel_oem7_corrimu/                             corrected IMU Measurements
        │    ├── [n  ]_[timestamp].csv                        file names correspond to data from other sensors
        │    ├── [n+1]_[timestamp].csv
        │    └── ...
        │
        ├── novatel_oem7_inspva/                              INS Position, Velocity and Attitude
        ├── novatel_oem7_insstdev/                            INS PVA standard deviations
        ├── radar/                                            images from radar sensor
        ├── readme_img/                                       illustrative images for readme
        ├── rgb_center/                                       images from the center 5 MP color camera
        ├── rgb_highres_center/                               images from the center 12 MP color camera
        ├── rgb_highres_left/                                 images from the left 12 MP color camera
        ├── rgb_highres_right/                                images from the right 12 MP color camera
        ├── rgb_left/                                         images from the left 5 MP color camera
        ├── rgb_right/                                        images from the right 5 MP color camera
        │
        ├── 1_calibration_1.1_labels.json                     annotation data for this sequence in ASAM OpenLABEL format
        ├── calibration.txt                                   calibration data for optical sensors (IR, RGB, Lidar, Radar)
        ├── license.md                                        license info for this sequence
        └── readme.md                                         this file

For information regarding the annotation data format, see the specification of [ASAM OpenLABEL](https://www.asam.net/standards/detail/openlabel/) and the [RailLabel](https://github.com/DSD-DBS/raillabel) package by DB Netz AG as well as the documentation included in the [Labeling Guide (10.48755/dzsf.230012.05)](https://doi.org/10.48755/dzsf.230012.05) (available from the [DZSF website](https://www.dzsf.bund.de/SharedDocs/Standardartikel/DZSF/Projekte/Projekt_70_Reale_Datensaetze.html)).



## Sensor Setup
The sensor setup that was used to acquire sensor data is described below. 

    sensor modality     devices  sensor model                    feature                value
    ------------------  -------  ------------------------------  ---------------------  ---------------------------------
    12 MP color camera        3  Teledyne GenieNano 5GigE C4040  sensor data            RGB images (8 Bit, PNG)
                                                                 resolution             4 112 × 2 504 px
                                                                 sampling frequency     10 Hz (synchronized)
                                                                 alignment              trident *
                                                                 data folders           rgb_highres_left, 
                                                                                          rgb_highres_center, 
                                                                                          rgb_highres_right
    
     5 MP color camera        3  Teledyne GenieNano C2420        sensor data            RGB images (8 Bit, PNG)
                                                                 resolution             2 464 × 1 600 px
                                                                 sampling frequency     10 Hz (synchronized)
                                                                 alignment              trident *
                                                                 data folders           rgb_left, rgb_center, rgb_right
     
    infrared camera           3  Teledyne Calibir DXM640         sensor data            grayscale images (8 Bit, PNG)
                                                                 resolution             640 × 480 px
                                                                 sampling frequency     10 Hz (synchronized)
                                                                 alignment              trident *
                                                                 data folders           ir_left, ir_center, ir_right

    long-range lidar          3  Livox Tele-15                   sensor data            3D point cloud (PCD)
                                                                 total sampling points  50 000 - 84 000 points per frame
                                                                 sampling frequency     10 Hz (synchronized)
                                                                 alignment              right, center, left
                                                                 data folders           lidar
                                                                 sensor index in pcd    1, 2, 3 (in order of alignment)
                                                                               
    medium-range lidar        1  HesaiTech Pandar64              sensor data            3D point cloud (PCD)
                                                                 total sampling points  60 000 - 115 000 points per frame
                                                                 sampling frequency     10 Hz (synchronized)
                                                                 alignment              central
                                                                 data folders           lidar
                                                                 sensor index in pcd    0

    short-range lidar         2  Waymo Honeycomb                 sensor data            3D point cloud (PCD)
                                                                 total sampling points  20 000 - 40 000 points per frame
                                                                 sampling frequency     10 Hz (synchronized)
                                                                 alignment              left, right
                                                                 data folders           lidar
                                                                 sensor index in pcd    4, 5 (in order of alignment)

    radar                     1  Navtech CIR204/H                sensor data            grayscale images (8 bit, PNG),
                                                                                          cartesian bird's eye view
                                                                 resolution             2 856 × 1 428 px
                                                                 sampling frequency     4 Hz (synchronized)
                                                                 data folders           radar

    localization and          1  NovAtel PwrPAk7D-E1             sensor data            linear and rotatory acceleration,
      inertial                                                                            latitude & longitude in WGS84
      measurement                                                sampling frequency     100/10 Hz
      units (IMU)                                                data folders           novatel_oem7_corrimu,
                                                                                          novatel_oem7_inspva,
                                                                                          novatel_oem7_insstdev


    * trident: three cameras are mounted in forward driving direction and oriented diagonal left, central and diagonal right

Data from all 6 lidar sensors are provided as merged point cloud ([Point Cloud Data (PCD) file format](https://pointclouds.org/documentation/tutorials/pcd_file_format.html)). Within the merged point cloud, sensors can be identified by the column `sensor_index` (see key above). Values for `intensity` are sensor-specific. 

The (merged) lidar coordinate system is the reference coordinate system. Its orientation is illustrated in the figure below.

![illustration of coordinate system](readme_img/fig_coordinate_system.png)
*Source: DB Netz AG*

Note that the extrinsic calibration parameters (`pose_wrt_parent` in the annotation file) describe the transformation (translation and rotation in quaternions) from the lidar (reference) coordinate system into the non-optical camera coordinate system. This *non-optical coordinate system* is aligned similarly to the *lidar coordinate system*:

* the positive x-axis points forwards
* the positive y-axis points to the left
* the positive z-axis points upwards

In contrast, the *optical camera coordinate system* is aligned similarly to the *image coordinate system*:

* the positive x-axis points to the right
* the positive y-axis points downwards
* the positive z-axis points forwards

To calculate the 2D image coordinates of a 3D lidar point, the following steps must be performed:

1. The lidar point is transformed into the non-optical camera coordinate system using the extrinsic calibration parameters (`pose_wrt_parent` in the annotation file).
2. The resulting 3D point is transformed into the optical camera coordinate system using the rotation quaternion (`x=0.5; y=-0.5; z=0.5; w=-0.5`). 
3. The resulting 3D point is transformed into the 2D image coordinate system using the internal calibration parameters (`intrinsics_pinhole` in the annotation file).



## Localization and IMU Data
Localization and IMU data files contain the following data.

    sensor (folder name)    data field              description                                                                     unit        coordinate system
    ----------------------  ----------------------  ------------------------------------------------------------------------------  ----------  -----------------
    novatel_oem7_corrimu                            https://docs.novatel.com/OEM7/Content/SPAN_Logs/CORRIMUS.htm
                            frame_idx               the frame's number this data belongs to
                            timestamp               at which time this sample was recorded                                          s
                            gps_week_number         number of weeks that passed GPS-start (1980-01-06T00:00:00Z)                    weeks
                            gps_week_milliseconds   number of milliseconds that passed since gps_week_number                        ms
                            pitch_rate              "velocity" of pitch: right-handed rotation around X-axis of the sensor frame    rad/sample  GNSS sensor
                            roll_rate               "velocity" of roll: right-handed rotation around Y-axis of the sensor frame     rad/sample  GNSS sensor
                            yaw_rate                "velocity" of yaw: right-handed rotation around Z-axis of the sensor frame      rad/sample  GNSS sensor
                            lateral_acc             lateral acceleration: along the sensor's X-axis                                 m/s/sample  GNSS sensor
                            longitudinal_acc        longitudinal acceleration: along the sensor's Y-axis                            m/s/sample  GNSS sensor
                            vertical_acc            vertical acceleration: along the sensor's Z-axis                                m/s/sample  GNSS sensor
    
    novatel_oem7_inspva                             https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSPVA.htm
                            frame_idx               see novatel_oem7_corrimu
                            timestamp               see novatel_oem7_corrimu
                            latitude                latitude in WGS84                                                               deg         earth (in WGS84)
                            longitude               longitude in WGS84                                                              deg         earth (in WGS84)
                            height                  ellipsodial/geodetic height                                                     m           earth (in WGS84)
                            north_velocity          velocity in northerly direction in ENU coordinates                              m/s         vehicle (in ENU)
                            east_velocity           velocity in easterly direction in ENU coordinates                               m/s         vehicle (in ENU)
                            up_velocity             velocity in an up direction in ENU coordinates                                  m/s         vehicle (in ENU)
                            roll                    right-handed rotation from local level around the sensor's Y-axis               deg         GNSS sensor
                            pitch                   right-handed rotation from local level around the sensor's X-axis               deg         GNSS sensor
                            azimuth                 left-handed rotation around the sensor's Z-axis clockwise from North            deg         GNSS sensor
    
    novatel_oem7_insstdev                           https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSSTDEV.htm
                            frame_idx               see novatel_oem7_corrimu
                            timestamp               see novatel_oem7_corrimu
                            latitude_stdev          standard deviation of inspva.latitude                                           m
                            longitude_stdev         standard deviation of inspva.longitude                                          m
                            height_stdev            standard deviation of inspva.height                                             m
                            north_velocity_stdev    standard deviation of inspva.north_velocity                                     m/s
                            east_velocity_stdev     standard deviation of inspva.east_velocity                                      m/s
                            up_velocity_stdev       standard deviation of inspva.up_velocity                                        m/s
                            roll_stdev              standard deviation of inspva.roll                                               deg
                            pitch_stdev             standard deviation of inspva.pitch                                              deg
                            azimuth_stdev           standard deviation of inspva.azimuth                                            deg

Point Cloud Data files contain the following data.

    row  starts with  description
    ---  -----------  -----------------------------------------------------------------------------------------
     1   # .PCD       file type info, see https://pointclouds.org/documentation/tutorials/pcd_file_format.html
     2   VERSION      specifies the PCD file version
     3   FIELDS       specifies the name of each dimension/field that a point can have
     4   SIZE         specifies the size of each dimension in bytes
     5   TYPE         specifies the type of each dimension as a char
                        I: signed types int8 (char), int16 (short), and int32 (int)
                        U: unsigned types uint8 (unsigned char), uint16 (unsigned short), uint32 (unsigned int)
                        F: represents float types
     5   COUNT        specifies how many elements each dimension has
     6   WIDTH        specifies the width of the point cloud dataset in the number of points
     7   HEIGHT       specifies the height of the point cloud dataset in the number of points
     8   VIEWPOINT    specifies an acquisition viewpoint for the points in the dataset
                        as a translation (tx ty tz) + quaternion (qw qx qy qz)
     9   POINTS       specifies the total number of points in the cloud
    10   DATA         specifies the data type that the point cloud data is stored in
                        three data types are supported: ascii, binary, and binary_compressed
    11+               data point with fields in columns according to the definition in FIELDS:
                        x y z intensity timestamp sensor_index
