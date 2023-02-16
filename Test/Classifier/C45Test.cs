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
            Assert.AreEqual(4.00, 100 * c45.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(bupa.GetInstanceList(), c45Parameter);
            Assert.AreEqual(36.82, 100 * c45.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(dermatology.GetInstanceList(), c45Parameter);
            Assert.AreEqual(2.46, 100 * c45.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(car.GetInstanceList(), c45Parameter);
            Assert.AreEqual(8.51, 100 * c45.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(tictactoe.GetInstanceList(), c45Parameter);
            Assert.AreEqual(11.06, 100 * c45.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            c45.Train(nursery.GetInstanceList(), c45Parameter);
            Assert.AreEqual(2.21, 100 * c45.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}