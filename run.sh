mvn package
bin/beast target/group1_projectC-1.0-SNAPSHOT.jar task1 wildfiredb_1k.csv.bz2
bin/beast target/group1_projectC-1.0-SNAPSHOT.jar task4 wildfiredb_ZIP
rm -rf wildfireIntensityCounty
# excecute command, the last command, data file, city file, begin time, end time
spark-submit \
--jars $(echo ~/jars/*.jar | tr ' ' ',') \
--class org.example.Task2 \
--master local \
--executor-memory 3G \
WildfireDataAnalysis-1.0-SNAPSHOT.jar \
wildfiredb_sample.parquet tl_2018_us_county.zip 01/01/2000 01/01/2024
