Name:Tianshi Liu
NetID:tliu222
Email:tliu222@ucr.edu
StudentID:X714911


The process to complete this task2

1.Start the Spark cluster.

2.Upload the required data files, primarily the city file tl_2018_us_county.zip, and the data file to be analyzed wildfiredb_sample.parquet.
    
3.Export all dependencies using Maven: Run mvn dependency:copy-dependencies -DincludeScope=runtime -DoutputDirectory=jars, then upload the jars directory to the machine where the Spark cluster is located.
   
4.Compile the project using Maven: Run mvn install, then upload the generated target/WildfireDataAnalysis-1.0-SNAPSHOT.jar to the machine where the Spark cluster is located.
    
5.Run the Spark job using a shell script (run.sh): Execute sh run.sh.
    
6.Download the entire wildfireIntensityCounty directory containing the results to your local computer.
    
7.Choose the part-00000.shp file from the wildfireIntensityCounty directory and import it into the QGIS program.
    
8.Adjust the relevant parameters in QGIS for further analysis and visualization.