package org.example

import edu.ucr.cs.bdlab.beast._
import org.apache.commons.lang3.time.DateUtils
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.sql.SparkSession

import java.text.SimpleDateFormat

object Task2 {

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
