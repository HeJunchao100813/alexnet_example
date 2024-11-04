from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

def main():
    # 初始化Spark会话
    spark = SparkSession.builder \
        .appName("Twitter Geo Clustering") \
        .master("local[*]") \
        .getOrCreate()

    # 加载推文数据集
    tweets = spark.read \
        .option("header", "true") \
        .csv("C:\\Users\\HeJunchao\\Desktop\\NUIG\\2425-CT5105-Java\\archive\\Tweets.csv") # 你的推文数据路径

    # 加载城市数据集并解析坐标
    cities = spark.read \
        .option("header", "true") \
        .option("delimiter", ";") \
        .csv("C:\\Users\\HeJunchao\\Desktop\\NUIG\\2425-CT5105-Java\\geonames-all-cities-with-a-population-1000.csv") # 你的城市数据路径

    # 解析Coordinates列，将其拆分为latitude和longitude
    cities = cities.withColumn("latitude", F.split(cities["Coordinates"], ",").getItem(0).cast(DoubleType())) \
                   .withColumn("longitude", F.split(cities["Coordinates"], ",").getItem(1).cast(DoubleType())) \
                   .withColumnRenamed("Name", "name") \
                   .select("name", "latitude", "longitude")

    # 预处理和转换数据
    preparedData = preprocess_and_transform_data(tweets, cities, spark)

    # 设置KMeans聚类，设置k=25
    kmeans = KMeans().setK(25).setSeed(1).setFeaturesCol("features").setPredictionCol("prediction")
    model = kmeans.fit(preparedData)

    # 评估聚类结果
    evaluator = ClusteringEvaluator(predictionCol="prediction")
    silhouette = evaluator.evaluate(model.transform(preparedData))
    print(f"Silhouette with squared Euclidean distance = {silhouette}")

    # 显示1000条推文的聚类结果
    predictions = model.transform(preparedData).limit(1000)
    predictions.select("text", "prediction").show()

    # 停止Spark会话
    spark.stop()

def preprocess_and_transform_data(tweets, cities, spark):
    # 选择推文中的必要列
    tweets = tweets.select("text", "tweet_coord", "tweet_location", "user_timezone")

    # 将推文与城市数据集连接，以获得缺失位置的坐标
    tweets_with_coords = tweets.join(cities, tweets["tweet_location"] == cities["name"], "left_outer") \
        .withColumn("latitude", F.coalesce(tweets["tweet_coord"].cast(DoubleType()), cities["latitude"])) \
        .withColumn("longitude", F.coalesce(tweets["tweet_coord"].cast(DoubleType()), cities["longitude"])) \
        .drop("name")  # 删除重复的城市名称列

    # 填充缺失的latitude和longitude为0.0，以避免聚类错误
    tweets_with_coords = tweets_with_coords.na.fill({"latitude": 0.0, "longitude": 0.0})

    # 将latitude和longitude组合成特征向量，以供K-means使用
    assembler = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="features")
    tweets_with_coords = assembler.transform(tweets_with_coords)

    # 返回包含必要列的DataFrame
    return tweets_with_coords.select("text", "features")

if __name__ == "__main__":
    main()
