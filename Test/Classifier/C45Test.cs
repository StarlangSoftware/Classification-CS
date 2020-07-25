using Classification.Classifier;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class C45Test : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var c45 = new C45();
            var c45Parameter = new C45Parameter(1, true, 0.2);
            c45.Train(iris.GetInstanceList(), c45Parameter);
            Assert.AreEqual(4.67, 100 * c45.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(bupa.GetInstanceList(), c45Parameter);
            Assert.AreEqual(42.03, 100 * c45.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(dermatology.GetInstanceList(), c45Parameter);
            Assert.AreEqual(6.28, 100 * c45.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(car.GetInstanceList(), c45Parameter);
            Assert.AreEqual(21.35, 100 * c45.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(tictactoe.GetInstanceList(), c45Parameter);
            Assert.AreEqual(19.94, 100 * c45.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(nursery.GetInstanceList(), c45Parameter);
            Assert.AreEqual(29.03, 100 * c45.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}