namespace TextAnalysis.Test.GeoInfo;

using global::GeoInfo;

[TestFixture]
public class GeoDataInfoTests {
	[Test]
	public void InfoIsAvailable() {
		Assert.That(GeoDataInfo.LastUpdated, Is.GreaterThan(DateTime.MinValue));
	}
}