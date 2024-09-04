using Classification.DistanceMetric;
using Classification.Filter;
using Classification.Model;
using Classification.Parameter;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Filter
{
    public class NormalizeTest : ClassifierTest
    {
        [Test]
        public void TestLinearPerceptron()
        {
            var linearPerceptron = new LinearPerceptronModel();
            var linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
            var normalize = new Normalize(iris);
            normalize.Convert();
            linearPerceptron.Train(iris.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(2.67, 100 * linearPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            normalize = new Normalize(bupa);
            normalize.Convert();
            linearPerceptron.Train(bupa.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(31.59, 100 * linearPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            normalize = new Normalize(dermatology);
            normalize.Convert();
            linearPerceptron.Train(dermatology.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(1.09, 100 * linearPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]
        public void TestMultiLayerPerceptron()
        {
            var multiLayerPerceptron = new MultiLayerPerceptronModel();
            var multiLayerPerceptronParameter =
                new MultiLayerPerceptronParameter(1, 1, 0.99, 0.2, 100, 3, ActivationFunction.SIGMOID);
            var normalize = new Normalize(iris);
            normalize.Convert();
            multiLayerPerceptron.Train(iris.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(3.33, 100 * multiLayerPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.5, 0.99, 0.2, 100, 30, ActivationFunction.SIGMOID);
            normalize = new Normalize(bupa);
            normalize.Convert();
            multiLayerPerceptron.Train(bupa.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(24.06, 100 * multiLayerPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 20, ActivationFunction.SIGMOID);
            normalize = new Normalize(dermatology);
            normalize.Convert();
            multiLayerPerceptron.Train(dermatology.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(1.09, 100 * multiLayerPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]
        public void TestKnn()
        {
            var knn = new KnnModel();
            var knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
            var normalize = new Normalize(iris);
            normalize.Convert();
            knn.Train(iris.GetInstanceList(), knnParameter);
            Assert.AreEqual(4.67, 100 * knn.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            normalize = new Normalize(bupa);
            normalize.Convert();
            knn.Train(bupa.GetInstanceList(), knnParameter);
            Assert.AreEqual(16.52, 100 * knn.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            normalize = new Normalize(dermatology);
            normalize.Convert();
            knn.Train(dermatology.GetInstanceList(), knnParameter);
            Assert.AreEqual(1.91, 100 * knn.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]
        public void TestKMeans()
        {
            var kMeans = new KMeansModel();
            var kMeansParameter = new KMeansParameter(1);
            var normalize = new Normalize(iris);
            normalize.Convert();
            kMeans.Train(iris.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(14.66, 100 * kMeans.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            normalize = new Normalize(bupa);
            normalize.Convert();
            kMeans.Train(bupa.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(41.44, 100 * kMeans.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            normalize = new Normalize(dermatology);
            normalize.Convert();
            kMeans.Train(dermatology.GetInstanceList(), kMeansParameter);
            Assert.AreEqual(3.55, 100 * kMeans.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}