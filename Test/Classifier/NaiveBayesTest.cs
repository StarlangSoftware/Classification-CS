using Classification.Model;
using NUnit.Framework;

namespace Test.Classifier
{
    public class NaiveBayesTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var naiveBayes = new NaiveBayesModel();
            naiveBayes.Train(iris.GetInstanceList(), null);
            Assert.AreEqual(5.33, 100 * naiveBayes.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            naiveBayes.Train(bupa.GetInstanceList(), null);
            Assert.AreEqual(38.55, 100 * naiveBayes.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            naiveBayes.Train(dermatology.GetInstanceList(), null);
            Assert.AreEqual(9.56, 100 * naiveBayes.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
            naiveBayes.Train(car.GetInstanceList(), null);
            Assert.AreEqual(12.91, 100 * naiveBayes.Test(car.GetInstanceList()).GetErrorRate(), 0.01);
            naiveBayes.Train(tictactoe.GetInstanceList(), null);
            Assert.AreEqual(30.17, 100 * naiveBayes.Test(tictactoe.GetInstanceList()).GetErrorRate(), 0.01);
            naiveBayes.Train(nursery.GetInstanceList(), null);
            Assert.AreEqual(9.70, 100 * naiveBayes.Test(nursery.GetInstanceList()).GetErrorRate(), 0.01);
        }
        
        [Test]
        public void TestLoad()
        {
            var naiveBayes = new NaiveBayesModel();
            naiveBayes.LoadModel("../../../models/naiveBayes-iris.txt");
            Assert.AreEqual(5.33, 100 * naiveBayes.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            naiveBayes.LoadModel("../../../models/naiveBayes-bupa.txt");
            Assert.AreEqual(38.55, 100 * naiveBayes.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            naiveBayes.LoadModel("../../../models/naiveBayes-dermatology.txt");
            Assert.AreEqual(9.56, 100 * naiveBayes.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }

    }
}