using Classification.Classifier;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class MultiLayerPerceptronTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var multiLayerPerceptron = new MultiLayerPerceptron();
            var multiLayerPerceptronParameter =
                new MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 3);
            multiLayerPerceptron.Train(iris.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(30.67, 100 * multiLayerPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 30);
            multiLayerPerceptron.Train(bupa.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(27.54, 100 * multiLayerPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 20);
            multiLayerPerceptron.Train(dermatology.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(4.92, 100 * multiLayerPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}