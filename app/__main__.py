from pathlib import Path
from typing import ClassVar
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.ml.feature import (
    VectorAssembler,
    BucketedRandomProjectionLSH,
    BucketedRandomProjectionLSHModel,
)
from pyspark.sql import DataFrame, Row


class Main:
    _PICTURE_DESCRIPTORS_FILE: ClassVar[Path] = Path(
        ".databases/descriptors/RESNETDescriptors.txt"
    )
    _PICTURES_DIR: ClassVar[Path] = Path(".databases/pictures")
    _GT_DIR: ClassVar[Path] = Path(".databases/ground_truth")
    _QUERIES: ClassVar[list[str]] = [
        "181081",
        "415008",
    ]

    @classmethod
    def run(cls) -> None:
        spark: SparkSession = cls.create_session()
        dataset: DataFrame = cls.load_dataset(spark)
        model: BucketedRandomProjectionLSHModel = cls.load_brp_model(dataset, 2.0, 3)
        dataset = cls.build_hashes(dataset, model)

        for picture_name in cls._QUERIES:
            df_result: DataFrame = cls.query(picture_name + ".jpg", dataset, model)
            print(f"Output for {picture_name}:")
            df_result.select("_c0", "distCol").show(10)
            pre, rec = cls.evaluate_precision_recall(
                df_result, cls._GT_DIR / (picture_name + ".txt")
            )
            print(f"P/R values for {picture_name}:", pre, rec)

    @classmethod
    def query(
        cls,
        picture_name: str,
        dataset: DataFrame,
        model: BucketedRandomProjectionLSHModel,
    ) -> DataFrame:
        query: DenseVector = cls.get_picture_embedding(dataset, picture_name)
        df_result: DataFrame = model.approxNearestNeighbors(dataset, query, 200)
        return df_result

    @classmethod
    def evaluate_precision_recall(
        cls, query_result: DataFrame, gt_file: Path
    ) -> tuple[float, float]:
        ground_truth: set[str] = set(gt_file.read_text("utf-8").splitlines())
        retrieved_images: set[str] = {row._c0 for row in query_result.collect()}
        true_positives: int = retrieved_images.intersection(ground_truth)
        precision: float = len(true_positives) / len(retrieved_images)
        recall: float = len(true_positives) / len(ground_truth)
        return precision, recall

    @classmethod
    def create_session(cls) -> SparkSession:
        session: SparkSession = (
            SparkSession.builder.master("local")
            .appName("BRPExampleOnRESTNET18")
            .config("spark.network.timeout", "600s")
            .config("spark.executor.heartbeatInterval", "120s")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("ERROR")
        return session

    @classmethod
    def load_dataset(cls, session: SparkSession) -> DataFrame:
        df: DataFrame = session.read.load(
            str(cls._PICTURE_DESCRIPTORS_FILE),
            format="csv",
            sep=" ",
            inferSchema="true",
            header="false",
        )
        assembler: VectorAssembler = VectorAssembler(
            inputCols=["_c" + str(i) for i in range(1, 513)], outputCol="features"
        )
        df = assembler.transform(df)
        return df

    @classmethod
    def load_brp_model(
        cls, dataset: DataFrame, bucket_size: float, num_hyperplanes: int
    ) -> BucketedRandomProjectionLSHModel:
        brp: BucketedRandomProjectionLSH = BucketedRandomProjectionLSH(
            inputCol="features",
            outputCol="hashes",
            bucketLength=bucket_size,
            numHashTables=num_hyperplanes,
        )
        return brp.fit(dataset)

    @classmethod
    def build_hashes(
        cls, dataset: DataFrame, model: BucketedRandomProjectionLSHModel
    ) -> DataFrame:
        return model.transform(dataset)

    @classmethod
    def get_picture_embedding(cls, dataset: DataFrame, pic_name: str) -> DenseVector:
        line: Row = dataset.filter(dataset._c0 == pic_name).collect()[0]
        return Vectors.dense(line[-2])


if __name__ == "__main__":
    Main.run()
