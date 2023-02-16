using Classification.Classifier;
using Classification.DistanceMetric;
using Classification.Filter;
using Classification.Parameter;
using NUnit.Framework;
using Test.Classifier;

namespace Test.Filter
{
    public class LaryToBinaryTest : ClassifierTest
    {
        [Test]
        public void TestKnn()
        {
            var knn = new Knn();
            var knnParameter = new KnnParameter(1, 3, new EuclidianDistance());
            var laryToBinary = new LaryToBinary(car);
            laryToBinary.Convert();
            knn.Train(car.GetInstanceList(), knnParameter);
            Assert.AreEqual(4.74, 100 * knn.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            laryToBinary = new LaryToBinary(tictactoe);
            laryToBinary.Convert();
            knn.Train(tictactoe.GetInstanceList(), knnParameter);
            Assert.AreEqual(5.64, 100 * knn.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
        }

        [Test]
        public void TestC45()
        {
            var c45 = new C45();
            var c45Parameter = new C45Parameter(1, true, 0.2);
            var laryToBinary = new LaryToBinary(car);
            laryToBinary.Convert();
            c45.Train(car.GetInstanceList(), c45Parameter);
            Assert.AreEqual(2.31, 100 * c45.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            laryToBinary = new LaryToBinary(tictactoe);
            laryToBinary.Convert();
            c45.Train(tictactoe.GetInstanceList(), c45Parameter);
            Assert.AreEqual(7.52, 100 * c45.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            laryToBinary = new LaryToBinary(nursery);
            laryToBinary.Convert();
            c45.Train(nursery.GetInstanceList(), c45Parameter);
            Assert.AreEqual(0.42, 100 * c45.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}