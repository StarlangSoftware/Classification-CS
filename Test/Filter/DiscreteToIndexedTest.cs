using Classification.Classifier;
using Classification.DistanceMetric;
using Classification.Filter;
using Classification.Parameter;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Filter
{
    public class DiscreteToIndexedTest : ClassifierTest
    {
        [Test]

        public void TestLinearPerceptron()
        {
            var linearPerceptron = new LinearPerceptron();
            var linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
            var discreteToIndexed = new DiscreteToIndexed(car);
            discreteToIndexed.Convert();
            linearPerceptron.Train(car.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(13.31, 100 * linearPerceptron.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            discreteToIndexed = new DiscreteToIndexed(tictactoe);
            discreteToIndexed.Convert();
            linearPerceptron.Train(tictactoe.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(19.31, 100 * linearPerceptron.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]

        public void TestKnn()
        {
            var knn = new Knn();
            var knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
            var discreteToIndexed = new DiscreteToIndexed(car);
            discreteToIndexed.Convert();
            knn.Train(car.GetInstanceList(), knnParameter);
            Assert.AreEqual(4.75, 100 * knn.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            discreteToIndexed = new DiscreteToIndexed(tictactoe);
            discreteToIndexed.Convert();
            knn.Train(tictactoe.GetInstanceList(), knnParameter);
            Assert.AreEqual(5.64, 100 * knn.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]

        public void TestC45()
        {
            var c45 = new C45();
            var c45Parameter = new C45Parameter(1, true, 0.2);
            var discreteToIndexed = new DiscreteToIndexed(car);
            discreteToIndexed.Convert();
            c45.Train(car.GetInstanceList(), c45Parameter);
            Assert.AreEqual(10.76, 100 * c45.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            discreteToIndexed = new DiscreteToIndexed(tictactoe);
            discreteToIndexed.Convert();
            c45.Train(tictactoe.GetInstanceList(), c45Parameter);
            Assert.AreEqual(16.08, 100 * c45.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            discreteToIndexed = new DiscreteToIndexed(nursery);
            discreteToIndexed.Convert();
            c45.Train(nursery.GetInstanceList(), c45Parameter);
            Assert.AreEqual(24.95, 100 * c45.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}