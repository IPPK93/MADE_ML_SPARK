package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.VectorAssembler
import breeze.linalg.{Vector => BreezeVector, DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector, *}
import org.apache.spark.sql.types.{StructField, DoubleType, StructType}
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.01
  lazy val data: DataFrame = LinearRegressionTest._data

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    
    val predictions: Array[Double] = data.collect().map(_.getAs[Double](2))
    val label: Array[Double] = data.collect().map(_.getAs[Double](1))

    model.weights.toArray.zip(LinearRegressionTest.trueW.toArray).foreach {
      case (w: Double, tw: Double) => w should be(tw +- delta)
    }

    model.bias should be(LinearRegressionTest.trueB +- delta)

    predictions.length should be(data.count())

    (predictions zip label).foreach {case (p: Double, l: Double) => p should be(l +- delta)}
  }

  "Estimator" should "calculate weights & bias" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("predictions")

    estimator.setStepSize(0.5)
    estimator.setMaxIter(200)

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }
  
  "Model" should "make predictions" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.dense(1.5, 0.3, -0.7).toDense,
      bias = 0.0
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("predictions")

    validateModel(model, model.transform(data))
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new ml.Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("predictions")
        .setStepSize(0.5)
        .setMaxIter(200)
    ))


    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite.save(tmpFolder.getAbsolutePath)

    val reRead = ml.Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModel(model, model.transform(data))
  }


  "Model" should "work after re-read" in {
    val pipeline = new ml.Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("predictions")
        .setStepSize(0.5)
        .setMaxIter(200)
    ))

    val model = pipeline.fit(data)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: ml.PipelineModel = ml.PipelineModel.load(tmpFolder.getAbsolutePath)
    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data))
  }


}

object LinearRegressionTest extends WithSpark {

  lazy val myMatrix = BreezeDenseMatrix.rand[Double](100000, 3)
  lazy val trueW = BreezeDenseVector(1.5, 0.3, -0.7)
  lazy val trueB = 0.0
  lazy val label = myMatrix * trueW +:+ trueB

  lazy val tempData = BreezeDenseMatrix.horzcat(myMatrix, label.toDenseMatrix.t)

  lazy val tempDataFull: DataFrame = sqlc.createDataFrame(tempData(*, ::)
    .iterator
    .map(x => (x(0), x(1), x(2), x(3)))
    .toSeq).toDF("x1", "x2", "x3", "label")

    lazy val _assembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3")).setOutputCol("features")
    lazy val _data: DataFrame = _assembler.transform(tempDataFull).select("features", "label")
}