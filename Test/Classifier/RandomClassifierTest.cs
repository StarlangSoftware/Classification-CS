using Classification.Model;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class RandomClassifierTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var randomClassifier = new RandomModel();
            var parameter = new Parameter(1);
            randomClassifier.Train(iris.GetInstanceList(), parameter);
            Assert.AreEqual(69.33, 100 * randomClassifier.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.Train(bupa.GetInstanceList(), parameter);
            Assert.AreEqual(51.59, 100 * randomClassifier.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.Train(dermatology.GetInstanceList(), parameter);
            Assert.AreEqual(84.97, 100 * randomClassifier.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.Train(car.GetInstanceList(), parameter);
            Assert.AreEqual(75.58, 100 * randomClassifier.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.Train(tictactoe.GetInstanceList(), parameter);
            Assert.AreEqual(46.35, 100 * randomClassifier.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.Train(nursery.GetInstanceList(), parameter);
            Assert.AreEqual(80.39, 100 * randomClassifier.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.Train(chess.GetInstanceList(), parameter);
            Assert.AreEqual(94.53, 100 * randomClassifier.Test(chess.GetInstanceList()).GetErrorRate(), 0.01);
        }
        
        [Test]
        public void TestLoad()
        {
            var randomClassifier = new RandomModel();
            randomClassifier.LoadModel("../../../models/random-iris.txt");
            Assert.AreEqual(69.33, 100 * randomClassifier.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.LoadModel("../../../models/random-bupa.txt");
            Assert.AreEqual(51.59, 100 * randomClassifier.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.LoadModel("../../../models/random-dermatology.txt");
            Assert.AreEqual(84.97, 100 * randomClassifier.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.LoadModel("../../../models/random-car.txt");
            Assert.AreEqual(75.58, 100 * randomClassifier.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.LoadModel("../../../models/random-tictactoe.txt");
            Assert.AreEqual(46.35, 100 * randomClassifier.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.LoadModel("../../../models/random-nursery.txt");
            Assert.AreEqual(80.39, 100 * randomClassifier.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
            randomClassifier.LoadModel("../../../models/random-chess.txt");
            Assert.AreEqual(94.53, 100 * randomClassifier.Test(chess.GetInstanceList()).GetErrorRate(), 0.01);
        }

    }
}