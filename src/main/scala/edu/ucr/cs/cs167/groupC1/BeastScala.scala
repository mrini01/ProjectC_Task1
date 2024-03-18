package edu.ucr.cs.cs167.groupC1
import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, functions => F}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, to_date, year, month, avg, sum}
import org.apache.spark.rdd.RDD
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.sql.SparkSession
import java.text.SimpleDateFormat
import org.apache.commons.lang3.time.DateUtils
import org.apache.spark.SparkConf


/**
 * Scala examples for Beast
 */
object BeastScala {
  def main(args: Array[String]): Unit = {
    // Initialize Spark context

    val conf = new SparkConf().setAppName("Wildfire Data Analysis")
    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")


    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()
    val sparkContext = sparkSession.sparkContext
    //CRSServer.startServer(sparkContext)
    SparkSQLRegistration.registerUDT
    SparkSQLRegistration.registerUDF(sparkSession)

    val operation: String = args(0)
    val inputFile: String = args(1)
    try {
      // Import Beast features
      import edu.ucr.cs.bdlab.beast._
      val t1 = System.nanoTime()
      var validOperation = true;

      //TASK 1
      operation match {
        case "task1" =>

          //1 Parse and load the CSV file using the Dataframe API.
          val wildfireDF = sparkSession.read.format("csv").option("delimiter", "\t").option("inferSchema", "true").option("header", "true").load(inputFile)

          //2,3,4,5,6: Keep only the following 13 columns, add geometry, change frp column to double, and cast numericals to double.Cast all numericals to double and save final dataframe as RDD
          val wildfireRDD: SpatialRDD = wildfireDF.selectExpr(
            "x",
            "y",
            "cast(acq_date as string) AS acq_date",
            "double(split(frp,',')[0]) AS frp",
            "acq_time",
            "cast(ELEV_mean as double) AS ELEV_mean",
            "cast(SLP_mean as double) AS SLP_mean",
            "cast(EVT_mean as double) AS EVT_mean",
            "cast(EVH_mean as double) AS EVH_mean",
            "cast(CH_mean as double) AS CH_mean",
            "cast(TEMP_ave as double) AS TEMP_ave",
            "cast(TEMP_min as double) AS TEMP_min",
            "cast(TEMP_max as double) AS TEMP_max",
            "ST_CreatePoint(x,y) AS geometry"
          ).toSpatialRDD

          //7 & 8. Load the County dataset using Beast. Run a spatial join query to find the county of each wildfire.
          val countiesDF = sparkSession.read.format("shapefile").load("tl_2018_us_county.zip")
          val countiesRDD: SpatialRDD = countiesDF.toSpatialRDD
          val wildfireCountyJoin: RDD[(IFeature, IFeature)] = wildfireRDD.spatialJoin(countiesRDD)

          //9 & 10.New attribute "County" via GEOID. Then convert it to dataframe.
          val wildfireCounty: DataFrame = wildfireCountyJoin.map({ case (wildfire, county) => Feature.append(wildfire, county.getAs[String]("GEOID"), "County") })
            .toDataFrame(sparkSession)

          //11. Complete dataframe without geometry column
          val completeDF: DataFrame = wildfireCounty.selectExpr(
            "x",
            "y",
            "cast(acq_date as string) AS acq_date",
            "double(split(frp,',')[0]) AS frp",
            "acq_time",
            "cast(ELEV_mean as double) AS ELEV_mean",
            "cast(SLP_mean as double) AS SLP_mean",
            "cast(EVT_mean as double) AS EVT_mean",
            "cast(EVH_mean as double) AS EVH_mean",
            "cast(CH_mean as double) AS CH_mean",
            "cast(TEMP_ave as double) AS TEMP_ave",
            "cast(TEMP_min as double) AS TEMP_min",
            "cast(TEMP_max as double) AS TEMP_max",
            "County"
          ).drop("geometry")

          //TESTING-
          //completeDF.printSchema()
          //completeDF.show()

          //12. Write the saved data as a parquet file.
          completeDF.write.mode(SaveMode.Overwrite).parquet("wildfiredb_ZIP")

          val t2 = System.nanoTime()
          println(s"Operation '$operation' on file '$inputFile' took ${(t2 - t1) * 1E-9} seconds")
        //finished task 1

        //TASK 2:
        case "task2" =>
          val t3 = System.nanoTime()
        def init(spark: SparkSession): Unit = {
    // Start the CRSServer and store the information in SparkConf
    val sparkContext = spark.sparkContext
    //    CRSServer.startServer(sparkContext)
    SparkSQLRegistration.registerUDT
    SparkSQLRegistration.registerUDF(spark)
  }

  def doTask(spark: SparkSession, dataPath: String, countryPath: String, startDate: String, endDate: String): Unit = {
    println(s"document file => ${dataPath}")
    var dataDF = spark.read.parquet(dataPath)
    dataDF.show()
    println(s"struct1: ${dataDF.dtypes.toSeq}")
    dataDF.createOrReplaceTempView("tmp_data")

    val newColumnStr = dataDF.columns.map(c => if (Seq("acq_date").contains(c)) s"to_date(${c}) $c" else c).mkString(",")
    dataDF = spark.sql(s"select ${newColumnStr} from tmp_data")
    println(s"struct2: ${dataDF.dtypes.toSeq}")
    dataDF.createOrReplaceTempView("tmp_data")
    spark.sql(s"select count(*) cnt,min(acq_date) min_date,max(acq_date) max_date from tmp_data").show()

    println(s"city file => ${countryPath}")
    val countiesDF = spark.sparkContext.shapefile(countryPath).toDataFrame(spark)
    countiesDF.show()
    countiesDF.createOrReplaceTempView("tmp_country")
    val sql =
      s"""
         |select b.GEOID,b.NAME,b.g,a.fire_intensity from (
         |  select County,sum(frp) fire_intensity from tmp_data where acq_date BETWEEN '${startDate}' AND '${endDate}' group by County
         |) a inner join tmp_country b on a.County = b.GEOID
         |""".stripMargin.trim
    println(s"excecute SQL => ${sql}")
    val result = spark.sql(sql)
    result.show()
    result.toSpatialRDD.coalesce(1).saveAsShapefile("wildfireIntensityCounty")
  }

  val simple = new SimpleDateFormat("yyyy-MM-dd")

  def formatDate(str: String): String = {
    val date = DateUtils.parseDate(str, Array("yyyy-MM-dd", "MM/dd/yyyy"): _*)
    simple.format(date)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Analysis")
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    println(s"parameter => ${args.toSeq}")
    val dataPath = args(0)
    val countryPath = args(1)
    val startDate = formatDate(args(2))
    val endDate = formatDate(args(3))
    init(spark)
    doTask(spark, dataPath, countryPath, startDate, endDate)
    spark.stop()
  }

}




        case "task4" =>
          // Start timing
          val task4Start = System.nanoTime()

          // Load the Parquet file produced by Task 1
          val wildfireDF = sparkSession.read.parquet(inputFile)

          // Print the schema and show a few rows to verify the content and types
          wildfireDF.printSchema()
          wildfireDF.show(5)

          // Filling null values with zero
          val filledWildfireDF = wildfireDF.na.fill(0)

          // Prepare the data by converting 'acq_date' to DateType and extracting year and month
          val dataWithTime = filledWildfireDF
            .withColumn("acq_date", to_date(col("acq_date"), "yyyy-MM-dd"))
            .withColumn("year", year(col("acq_date")))
            .withColumn("month", month(col("acq_date")))

          // Group by county, year, and month, and aggregate
          val aggregatedData = dataWithTime
            .groupBy("County", "year", "month")
            .agg(
              sum("frp").alias("fire_intensity"),
              // Aggregate other columns by average
              avg("ELEV_mean").alias("ELEV_mean"),
              avg("SLP_mean").alias("SLP_mean"),
              avg("EVT_mean").alias("EVT_mean"),
              avg("EVH_mean").alias("EVH_mean"),
              avg("CH_mean").alias("CH_mean"),
              avg("TEMP_ave").alias("TEMP_ave"),
              avg("TEMP_min").alias("TEMP_min"),
              avg("TEMP_max").alias("TEMP_max")
            )

          // Features for the model
          val featureCols = Array("ELEV_mean", "SLP_mean", "EVT_mean", "EVH_mean", "CH_mean", "TEMP_ave", "TEMP_min", "TEMP_max")
          val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
          val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)

          // Initialize the Linear Regression model
          val lr = new LinearRegression().setLabelCol("fire_intensity").setFeaturesCol("scaledFeatures")

          // Create a Pipeline
          val pipeline = new Pipeline().setStages(Array(assembler, scaler, lr))

          // Split the data into training and test sets
          val Array(trainingData, testData) = aggregatedData.randomSplit(Array(0.8, 0.2))

          // Train the model
          val model = pipeline.fit(trainingData)

          // Make predictions
          val predictions = model.transform(testData)

          // Show schema and predictions
          predictions.select("ELEV_mean", "SLP_mean", "EVT_mean", "EVH_mean", "CH_mean", "TEMP_ave", "TEMP_min", "TEMP_max", "fire_intensity", "prediction").show()

          // End timing
          val task4End = System.nanoTime()

          // Compute total time
          val task4Time = (task4End - task4Start) / 1e9d
          println(s"Task 4 total time: $task4Time seconds")

          // Compute RMSE
          val evaluator = new RegressionEvaluator()
            .setLabelCol("fire_intensity")
            .setPredictionCol("prediction")
            .setMetricName("rmse")

          val rmse = evaluator.evaluate(predictions)
          println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

      }
    } finally {
      sparkSession.stop()
    }
  }
}
