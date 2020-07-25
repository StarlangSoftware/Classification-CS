using System.Collections.Generic;
using Classification.Attribute;
using Classification.DataSet;
using NUnit.Framework;

namespace Test.Classifier
{
    public class ClassifierTest
    {
        protected Classification.DataSet.DataSet iris, car, chess, bupa, tictactoe, dermatology, nursery;

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
    }
}