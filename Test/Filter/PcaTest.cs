using Classification.Classifier;
using Classification.DistanceMetric;
using Classification.Filter;
using Classification.Parameter;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Filter
{
    public class PcaTest : ClassifierTest
    {
        [Test]
        public void TestLinearPerceptron()
        {
            var linearPerceptron = new LinearPerceptron();
            var linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
            var pca = new Pca(iris);
            pca.Convert();
            linearPerceptron.Train(iris.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(2.67, 100 * linearPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            linearPerceptronParameter = new LinearPerceptronParameter(1, 0.01, 0.99, 0.2, 100);
            pca = new Pca(bupa);
            pca.Convert();
            linearPerceptron.Train(bupa.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(42.32, 100 * linearPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            pca = new Pca(dermatology);
            pca.Convert();
            linearPerceptron.Train(dermatology.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(3.00, 100 * linearPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]
        public void TestKnn()
        {
            var knn = new Knn();
            var knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
            var pca = new Pca(iris);
            pca.Convert();
            knn.Train(iris.GetInstanceList(), knnParameter);
            Assert.AreEqual(4.00, 100 * knn.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            pca = new Pca(bupa);
            pca.Convert();
            knn.Train(bupa.GetInstanceList(), knnParameter);
            Assert.AreEqual(19.13, 100 * knn.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            pca = new Pca(dermatology);
            pca.Convert();
            knn.Train(dermatology.GetInstanceList(), knnParameter);
            Assert.AreEqual(3.28, 100 * knn.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}