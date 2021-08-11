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
                new MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 3, ActivationFunction.SIGMOID);
            multiLayerPerceptron.Train(iris.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(5.33, 100 * multiLayerPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 30, ActivationFunction.SIGMOID);
            multiLayerPerceptron.Train(bupa.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(28.69, 100 * multiLayerPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 20, ActivationFunction.SIGMOID);
            multiLayerPerceptron.Train(dermatology.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(1.91, 100 * multiLayerPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}