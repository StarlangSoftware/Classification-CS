using Classification.Classifier;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class RandomForestTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var randomForest = new RandomForest();
            var randomForestParameter = new RandomForestParameter(1, 100, 35);
            randomForest.Train(iris.GetInstanceList(), randomForestParameter);
            Assert.AreEqual(0.0, 100 * randomForest.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            randomForest.Train(bupa.GetInstanceList(), randomForestParameter);
            Assert.AreEqual(0.0, 100 * randomForest.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            randomForest.Train(dermatology.GetInstanceList(), randomForestParameter);
            Assert.AreEqual(0.0, 100 * randomForest.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            randomForest.Train(car.GetInstanceList(), randomForestParameter);
            Assert.AreEqual(0.0, 100 * randomForest.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            randomForest.Train(tictactoe.GetInstanceList(), randomForestParameter);
            Assert.AreEqual(0.0, 100 * randomForest.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            randomForest.Train(nursery.GetInstanceList(), randomForestParameter);
            Assert.AreEqual(0.0, 100 * randomForest.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}