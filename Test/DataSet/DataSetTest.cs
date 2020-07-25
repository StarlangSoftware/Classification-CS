using System.Collections.Generic;
using Classification.Attribute;
using Classification.DataSet;
using NUnit.Framework;

namespace Test.DataSet
{
    public class DataSetTest
    {
        Classification.DataSet.DataSet iris, car, chess, bupa, tictactoe, dermatology, nursery;

        [SetUp]
        public void SetUp()
        {
            var attributeTypes = new List<AttributeType>();

            for (var i = 0; i < 4; i++)
            {
                attributeTypes.Add(AttributeType.CONTINUOUS);
            }

            var dataDefinition = new DataDefinition(attributeTypes);
            iris = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/iris.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 6; i++)
            {
                attributeTypes.Add(AttributeType.CONTINUOUS);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            bupa = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/bupa.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 34; i++)
            {
                attributeTypes.Add(AttributeType.CONTINUOUS);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            dermatology = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/dermatology.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 6; i++)
            {
                attributeTypes.Add(AttributeType.DISCRETE);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            car = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/car.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 9; i++)
            {
                attributeTypes.Add(AttributeType.DISCRETE);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            tictactoe = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/tictactoe.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 8; i++)
            {
                attributeTypes.Add(AttributeType.DISCRETE);
            }

            dataDefinition = new DataDefinition(attributeTypes);
            nursery = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/nursery.data");
            attributeTypes = new List<AttributeType>();
            for (var i = 0; i < 6; i++)
            {
                if (i % 2 == 0)
                {
                    attributeTypes.Add(AttributeType.DISCRETE);
                }
                else
                {
                    attributeTypes.Add(AttributeType.CONTINUOUS);
                }
            }

            dataDefinition = new DataDefinition(attributeTypes);
            chess = new Classification.DataSet.DataSet(dataDefinition, ",", "../../../datasets/chess.data");
        }

        [Test]
        public void TestSampleSize()
        {
            Assert.AreEqual(150, iris.SampleSize());
            Assert.AreEqual(345, bupa.SampleSize());
            Assert.AreEqual(366, dermatology.SampleSize());
            Assert.AreEqual(1728, car.SampleSize());
            Assert.AreEqual(958, tictactoe.SampleSize());
            Assert.AreEqual(12960, nursery.SampleSize());
            Assert.AreEqual(28056, chess.SampleSize());
        }

        [Test]
        public void TestClassCount()
        {
            Assert.AreEqual(3, iris.ClassCount());
            Assert.AreEqual(2, bupa.ClassCount());
            Assert.AreEqual(6, dermatology.ClassCount());
            Assert.AreEqual(4, car.ClassCount());
            Assert.AreEqual(2, tictactoe.ClassCount());
            Assert.AreEqual(5, nursery.ClassCount());
            Assert.AreEqual(18, chess.ClassCount());
        }

        [Test]
        public void TestGetClasses()
        {
            Assert.AreEqual("Iris-setosa;Iris-versicolor;Iris-virginica", iris.GetClasses());
            Assert.AreEqual("1;2", bupa.GetClasses());
            Assert.AreEqual("2;1;3;5;4;6", dermatology.GetClasses());
            Assert.AreEqual("unacc;acc;vgood;good", car.GetClasses());
            Assert.AreEqual("positive;negative", tictactoe.GetClasses());
            Assert.AreEqual("recommend;priority;not_recom;very_recom;spec_prior", nursery.GetClasses());
            Assert.AreEqual(
                "draw;zero;one;two;three;four;five;six;seven;eight;nine;ten;eleven;twelve;thirteen;fourteen;fifteen;sixteen"
                , chess.GetClasses());
        }
    }
}