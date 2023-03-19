name := "made_example"

version := "0.1"

scalaVersion := "2.12.17"

val sparkVersion = "3.0.1"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion withSources(),
  "org.apache.spark" %% "spark-mllib" % sparkVersion withSources(),
  "org.scalanlp" %% "breeze" % "1.3"
)

libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.2" % "test" withSources())