namespace TextAnalysis.Test.WebTranslator;

using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Mime;
using System.Security.Cryptography.Xml;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;

public class BasicTests {
	private WebApplicationFactory<Program> _factory;

	[SetUp]
	public void SetUp() {
		_factory = new CustomWebApplicationFactory<Program>();
	}

	[TearDown]
	public void TearDown() {
		_factory.Dispose();
	}

	[Test]
	public async Task IsUp() {
		using HttpClient client = _factory.CreateClient();
		using HttpResponseMessage response = await client.GetAsync("/Check");
		response.EnsureSuccessStatusCode();
		using HttpResponseMessage responseLowercase = await client.GetAsync("/check");
		responseLowercase.EnsureSuccessStatusCode();
	}

	[Test]
	public async Task CanTranslateDeToEn() {
		using HttpClient client = _factory.CreateClient();
		var dataObject = new { };
		String data = JsonSerializer.Serialize(dataObject);
		using HttpResponseMessage response = await client.PostAsync("/Translation", new StringContent(data, Encoding.UTF8, MediaTypeNames.Application.Json));
		String responseContent = await response.Content.ReadAsStringAsync();
		Console.WriteLine(responseContent);
		response.EnsureSuccessStatusCode();
	}

	[TestCase(null, null, HttpStatusCode.OK)]
	[TestCase("", null, HttpStatusCode.OK)]
	[TestCase("*/*", null, HttpStatusCode.OK)]
	[TestCase("application/json", null, HttpStatusCode.OK)]
	[TestCase("text/plain,application/json", null, HttpStatusCode.OK)]
	[TestCase("application/json", "application/json; charset=utf-16", HttpStatusCode.UnsupportedMediaType)]
	[TestCase("application/json", "application/json; charset=utf-8", HttpStatusCode.OK)]
	[TestCase("application/octet-stream", null, HttpStatusCode.NotAcceptable)]
	public async Task CanTranslateDeToEnWithAccept(String? accept, String? postContentType, HttpStatusCode expectedCode) {
		using HttpClient client = _factory.CreateClient();
		dynamic dataObject = new { from = "deu", to = "en-US", text = "Irgendein text.", ignored = new { whatever = 5 } };
		String data = JsonSerializer.Serialize(dataObject);

		using HttpRequestMessage request = new(HttpMethod.Post, "/Translation");
		if (postContentType != null)
			request.Content = new StringContent(data, MediaTypeHeaderValue.Parse(postContentType));
		else
			request.Content = new StringContent(data, Encoding.UTF8, MediaTypeNames.Application.Json);
		if (accept != null)
			request.Headers.Accept.ParseAdd(accept);

		using HttpResponseMessage response = await client.SendAsync(request);
		String responseContent = await response.Content.ReadAsStringAsync();
		Console.WriteLine(responseContent);
		response.StatusCode.Should().Be(expectedCode);
	}
}