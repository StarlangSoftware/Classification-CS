using Classification.Model;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class MultiLayerPerceptronTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var multiLayerPerceptron = new MultiLayerPerceptronModel();
            var multiLayerPerceptronParameter =
                new MultiLayerPerceptronParameter(1, 0.1, 0.99, 0.2, 100, 3, ActivationFunction.SIGMOID);
            multiLayerPerceptron.Train(iris.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(1.33, 100 * multiLayerPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 30, ActivationFunction.SIGMOID);
            multiLayerPerceptron.Train(bupa.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(31.30, 100 * multiLayerPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptronParameter = new MultiLayerPerceptronParameter(1, 0.01, 0.99, 0.2, 100, 20, ActivationFunction.SIGMOID);
            multiLayerPerceptron.Train(dermatology.GetInstanceList(), multiLayerPerceptronParameter);
            Assert.AreEqual(4.37, 100 * multiLayerPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
        
        [Test]
        public void TestLoad()
        {
            var multiLayerPerceptron = new MultiLayerPerceptronModel();
            multiLayerPerceptron.LoadModel("../../../models/multiLayerPerceptron-iris.txt");
            Assert.AreEqual(2.67, 100 * multiLayerPerceptron.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptron.LoadModel("../../../models/multiLayerPerceptron-bupa.txt");
            Assert.AreEqual(27.54, 100 * multiLayerPerceptron.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            multiLayerPerceptron.LoadModel("../../../models/multiLayerPerceptron-dermatology.txt");
            Assert.AreEqual(1.09, 100 * multiLayerPerceptron.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }

    }
}