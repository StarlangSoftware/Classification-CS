using Classification.Classifier;
using Classification.DistanceMetric;
using Classification.Filter;
using Classification.Parameter;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Filter
{
    public class DiscreteToContinuousTest : ClassifierTest
    {
        [Test]
        public void TestLinearPerceptron()
        {
            var linearPerceptron = new LinearPerceptron();
            var linearPerceptronParameter = new LinearPerceptronParameter(1, 0.1, 0.99, 0.2, 100);
            var discreteToContinuous = new DiscreteToContinuous(car);
            discreteToContinuous.Convert();
            linearPerceptron.Train(car.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(8.80, 100 * linearPerceptron.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            discreteToContinuous = new DiscreteToContinuous(tictactoe);
            discreteToContinuous.Convert();
            linearPerceptron.Train(tictactoe.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(1.67, 100 * linearPerceptron.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            discreteToContinuous = new DiscreteToContinuous(nursery);
            discreteToContinuous.Convert();
            linearPerceptron.Train(nursery.GetInstanceList(), linearPerceptronParameter);
            Assert.AreEqual(11.45, 100 * linearPerceptron.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]
        public void TestKnn()
        {
            var knn = new Knn();
            var knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
            var discreteToContinuous = new DiscreteToContinuous(car);
            discreteToContinuous.Convert();
            knn.Train(car.GetInstanceList(), knnParameter);
            Assert.AreEqual(4.75, 100 * knn.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            discreteToContinuous = new DiscreteToContinuous(tictactoe);
            discreteToContinuous.Convert();
            knn.Train(tictactoe.GetInstanceList(), knnParameter);
            Assert.AreEqual(5.64, 100 * knn.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]
        public void TestC45()
        {
            var c45 = new C45();
            var c45Parameter = new C45Parameter(1, true, 0.2);
            var discreteToContinuous = new DiscreteToContinuous(car);
            discreteToContinuous.Convert();
            c45.Train(car.GetInstanceList(), c45Parameter);
            Assert.AreEqual(29.98, 100 * c45.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            discreteToContinuous = new DiscreteToContinuous(tictactoe);
            discreteToContinuous.Convert();
            c45.Train(tictactoe.GetInstanceList(), c45Parameter);
            Assert.AreEqual(34.66, 100 * c45.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}