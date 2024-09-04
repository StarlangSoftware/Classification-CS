using Classification.Model;
using NUnit.Framework;

namespace Test.Classifier
{
    public class QdaTest : ClassifierTest
    {
        [Test]
        public void TestTrain()
        {
            var qda = new QdaModel();
            qda.Train(iris.GetInstanceList(), null);
            Assert.AreEqual(2.00, 100 * qda.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            qda.Train(bupa.GetInstanceList(), null);
            Assert.AreEqual(36.52, 100 * qda.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
        }
        
        [Test]
        public void TestLoad()
        {
            var qda = new QdaModel();
            qda.LoadModel("../../../models/qda-iris.txt");
            Assert.AreEqual(2.00, 100 * qda.Test(iris.GetInstanceList()).GetErrorRate(), 0.01);
            qda.LoadModel("../../../models/qda-bupa.txt");
            Assert.AreEqual(36.52, 100 * qda.Test(bupa.GetInstanceList()).GetErrorRate(), 0.01);
        }

    }
}