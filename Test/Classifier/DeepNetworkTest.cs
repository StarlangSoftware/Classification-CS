using System.Collections.Generic;
using Classification.Model;
using Classification.Parameter;
using NUnit.Framework;

namespace Test.Classifier
{
    public class DeepNetworkTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var deepNetwork = new DeepNetworkModel();

            var deepNetworkParameter =
                new DeepNetworkParameter(1, 0.1, 0.99, 0.2, 100, new List<int> {5, 5}, ActivationFunction.SIGMOID);

            deepNetwork.Train(iris.GetInstanceList(), deepNetworkParameter);
            Assert.AreEqual(2.67, 100 * deepNetwork.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            deepNetworkParameter = new DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, new List<int>
                {15, 15}, ActivationFunction.SIGMOID);
            deepNetwork.Train(bupa.GetInstanceList(), deepNetworkParameter);
            Assert.AreEqual(30.72, 100 * deepNetwork.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            deepNetworkParameter = new DeepNetworkParameter(1, 0.01, 0.99, 0.2, 100, new List<int>
                {20}, ActivationFunction.SIGMOID);
            deepNetwork.Train(dermatology.GetInstanceList(), deepNetworkParameter);
            Assert.AreEqual(4.09, 100 * deepNetwork.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }
        
        [Test]
        public void TestLoad()
        {
            var deepNetwork = new DeepNetworkModel();
            deepNetwork.LoadModel("../../../models/deepNetwork-iris.txt");
            Assert.AreEqual(1.33, 100 * deepNetwork.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            deepNetwork.LoadModel("../../../models/deepNetwork-bupa.txt");
            Assert.AreEqual(28.99, 100 * deepNetwork.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
            deepNetwork.LoadModel("../../../models/deepNetwork-dermatology.txt");
            Assert.AreEqual(1.09, 100 * deepNetwork.Test(dermatology.GetInstanceList()).GetErrorRate(), 0.01);
        }

    }
}