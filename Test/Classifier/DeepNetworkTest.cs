using System.Collections.Generic;
using Classification.Classifier;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class DeepNetworkTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var deepNetwork = new DeepNetwork();

            var deepNetworkParameter =
                new DeepNetworkParameter(1, 0.1, 0.99, 0.2, 100, new List<int> {5, 5}, ActivationFunction.SIGMOID);

            deepNetwork.Train(iris.GetInstanceList(), deepNetworkParameter);
            Assert.AreEqual(2.67, 100 * deepNetwork.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            deepNetworkParameter = new DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, new List<int>
                {15, 15}, ActivationFunction.SIGMOID);
            deepNetwork.Train(bupa.GetInstanceList(), deepNetworkParameter);
            Assert.AreEqual(29.57, 100 * deepNetwork.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            deepNetworkParameter = new DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, new List<int>
                {20}, ActivationFunction.SIGMOID);
            deepNetwork.Train(dermatology.GetInstanceList(), deepNetworkParameter);
            Assert.AreEqual(2.19, 100 * deepNetwork.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
    }
}