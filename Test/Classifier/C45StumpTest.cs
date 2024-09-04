using Classification.Model.DecisionTree;
using NUnit.Framework;

namespace Test.Classifier
{
    public class C45StumpTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var c45Stump = new DecisionStump();
            c45Stump.Train(iris.GetInstanceList(), null);
            Assert.AreEqual(33.33, 100 * c45Stump.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            c45Stump.Train(bupa.GetInstanceList(), null);
            Assert.AreEqual(36.81, 100 * c45Stump.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            c45Stump.Train(dermatology.GetInstanceList(), null);
            Assert.AreEqual(49.73, 100 * c45Stump.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            c45Stump.Train(car.GetInstanceList(), null);
            Assert.AreEqual(29.98, 100 * c45Stump.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            c45Stump.Train(tictactoe.GetInstanceList(), null);
            Assert.AreEqual(30.06, 100 * c45Stump.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            c45Stump.Train(nursery.GetInstanceList(), null);
            Assert.AreEqual(29.03, 100 * c45Stump.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
            c45Stump.Train(chess.GetInstanceList(), null);
            Assert.AreEqual(80.92, 100 * c45Stump.Test(chess.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}