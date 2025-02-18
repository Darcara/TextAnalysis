namespace TextAnalysis.Test.Generators;

using System.Collections.Frozen;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO.Compression;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Xml.Serialization;
using CsvHelper;
using CsvHelper.Configuration;
using global::GeoInfo.Iso4217;
using global::GeoInfo.Iso639;

[Category("Benchmark")]
[TestFixture]
public partial class GeonamesGenerator {
	private List<CurrencyEntry> _currencies;
	private List<Iso639Entry> Iso639Entries { get; set; } = null!;
	private List<Iso639NameEntry> Iso639NameEntries { get; set; } = null!;

	private List<DatasetsCountryCodesEntry> DatasetsCountryCodesEntries { get; set; } = null!;
	private List<GeonamesCountryInfoEntry> GeonamesCountryInfoEntries { get; set; } = null!;

	[OneTimeSetUp]
	public async Task PrepareData() {
		await DownloadData();
		GenerateLanguages();
		GenerateCountries();
		GenerateCurrencies();
	}

	private async Task DownloadData() {
		using HttpClient client = new();

		// await Helper.DownloadFile(client, "https://download.geonames.org/export/dump/allCountries.zip", "data/geo/allCountries.zip");
		// await Helper.DownloadFile(client, "https://download.geonames.org/export/dump/alternateNamesV2.zip", "data/geo/alternateNamesV2.zip");
		await Helper.DownloadFile(client, "https://download.geonames.org/export/dump/countryInfo.txt", "data/geo/countryInfo.txt");
		await Helper.DownloadFile(client, "https://download.geonames.org/export/dump/hierarchy.zip", "data/geo/hierarchy.zip");
		await Helper.DownloadFile(client, "https://download.geonames.org/export/dump/adminCode5.zip", "data/geo/adminCode5.zip");
		await Helper.DownloadFile(client, "https://download.geonames.org/export/dump/shapes_all_low.zip", "data/geo/countryShapes.zip");
		await Helper.DownloadFile(client, "https://download.geonames.org/export/dump/shapes_simplified_low.json.zip", "data/geo/countryShapesSimplified.zip");
		await Helper.DownloadFile(client, "https://download.geonames.org/export/dump/iso-languagecodes.txt", "data/geo/isoLanguageCodes.zip");

		// https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/
		await Helper.DownloadFile(client, "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip", "data/geo/countryBoundaries.zip");
		await Helper.DownloadFile(client, "https://raw.githubusercontent.com/datasets/country-codes/refs/heads/main/data/country-codes.csv", "data/geo/countryCodes.csv");
		await Helper.DownloadFile(client, "https://raw.githubusercontent.com/unicode-org/cldr/refs/heads/main/common/supplemental/supplementalData.xml", "data/geo/unicodeSupplementalData.xml");

		await Helper.DownloadFile(client, "https://www.loc.gov/standards/iso639-2/ISO-639-2_utf-8.txt", "data/geo/ISO-639-2_utf-8.txt");
		// See here for new Link https://iso639-3.sil.org/code_tables/download_tables#Complete%20Code%20Tables
		await Helper.DownloadFile(client, "https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3_Code_Tables_20241010.zip", "data/geo/ISO-639-3.zip");
		await Helper.DownloadFile(client, "https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml", "data/geo/ISO-4217-currency.xml");
	}

	private void GenerateCountries() {
		CsvConfiguration countryCodesConfig = new(CultureInfo.InvariantCulture) {
			MemberTypes = MemberTypes.Properties | MemberTypes.Fields,
			HasHeaderRecord = true,
			PrepareHeaderForMatch = args => Regex.Replace(args.Header, @"[ \-\(\)]+", "_").TrimEnd('_'),
		};
		using (CsvReader csvReader = new(File.OpenText("data/geo/countryCodes.csv"), countryCodesConfig, leaveOpen: false)) {
			DatasetsCountryCodesEntries = csvReader.GetRecords<DatasetsCountryCodesEntry>().ToList();
		}

		CsvConfiguration countryInfoConfig = new(CultureInfo.InvariantCulture) {
			MemberTypes = MemberTypes.Properties | MemberTypes.Fields,
			HasHeaderRecord = true,
			Comment = '#',
			AllowComments = true,
			Delimiter = "\t",
			PrepareHeaderForMatch = args => Regex.Replace(args.Header, @"[ \-\(\)]+", "_").TrimEnd('_'),
		};

		String countryInfoData = File.ReadAllText("data/geo/countryInfo.txt").Replace("\n#ISO\tISO3", "\nISO\tISO3");
		using (CsvReader csvReader = new(new StringReader(countryInfoData), countryInfoConfig, leaveOpen: false)) {
			GeonamesCountryInfoEntries = csvReader.GetRecords<GeonamesCountryInfoEntry>().ToList();
		}
	}

	private void GenerateCurrencies() {
		using StreamReader streamReader = File.OpenText("data/geo/ISO-4217-currency.xml");
		XmlSerializer serializer = new XmlSerializer(typeof(ISO4217));
		ISO4217? currencies = (ISO4217?)serializer.Deserialize(streamReader);
		if (currencies == null) return;

		_currencies = currencies.CcyTbl.CcyNtry;
	}

	[Test]
	public void Generate4217Currencies() {
		StringBuilder sb = new();

		sb.AppendLine("// <auto-generated>");
		sb.AppendLine("//   This file was generated by a tool; you should avoid making direct changes.");
		sb.AppendLine("// </auto-generated>");
		sb.AppendLine();
		sb.AppendLine("namespace GeoInfo.Iso4217;");
		sb.AppendLine("#region Designer generated code");
		sb.AppendLine("public enum Currency {");
		sb.AppendLine("///<summary>Not a currency.</summary>");
		sb.AppendLine("NotACurrency=-1,");
		sb.AppendLine("///<summary>Not a currency, but instead an uninitialized variable.</summary>");
		sb.AppendLine("Uninitialized=0,");
		Int32 numCurrencies = 0;

		FrozenSet<String> codesToIgnore = [
			// US Dollar (Next day),
			"USN",
			// SDR (Special Drawing Right)
			"XDR",
			// Bond Markets Unit European Composite Unit (EURCO)
			"XBA",
			// Bond Markets Unit European Monetary Unit (E.M.U.-6)
			"XBB",
			// Bond Markets Unit European Unit of Account 9 (E.U.A.-9)
			"XBC",
			// Bond Markets Unit European Unit of Account 17 (E.U.A.-17)
			"XBD",
			// ADB Unit of Account
			"XUA",
		];

		String GetCurrencyName(CurrencyEntry currencyEntry) {
			String name = Regex.Replace(currencyEntry.CcyNm.Text, @"[-’ \.]+", "");
			name = Regex.Replace(name, @"\([A-Z]+\)$", "", RegexOptions.Singleline);

			if (currencyEntry.Ccy == "XTS")
				name = "TestCurrency";
			if (currencyEntry.Ccy == "XXX")
				name = "NoCurrencyInvolved";
			return RemoveDiacritics(name);
		}

		foreach (IGrouping<String, CurrencyEntry> currencyEntries in _currencies.DistinctBy(cur => cur.Ccy).GroupBy(GetCurrencyName)) {
			Boolean hasMultiple = currencyEntries.Count() > 1;
			foreach (CurrencyEntry currencyEntry in currencyEntries) {
				if (currencyEntry.Ccy == null || codesToIgnore.Contains(currencyEntry.Ccy.ToUpperInvariant()))
					continue;

				numCurrencies++;

				sb.AppendLine("/// <summary>");
				sb.AppendLine($"/// <para>{currencyEntry.CcyNm.Text}</para>");
				sb.AppendLine("/// </summary>");
				sb.AppendLine($"/// <value>id={currencyEntry.Ccy}, numeric={currencyEntry.CcyNbr}</value>");

				sb.AppendLine($"{GetCurrencyName(currencyEntry)}{(hasMultiple?$"_{currencyEntry.Ccy}":String.Empty)}={CalculateFrom3And2Code(currencyEntry.Ccy, null)},");
			}
		}


		sb.AppendLine("}");
		sb.AppendLine("#endregion");
		Console.WriteLine(sb.ToString().Trim());
		Assert.Pass($"{numCurrencies} currencies created.");
	}

	[Test]
	public void Generate3166CountryEnum() {
		StringBuilder sb = new();
		sb.AppendLine("// <auto-generated>");
		sb.AppendLine("//   This file was generated by a tool; you should avoid making direct changes.");
		sb.AppendLine("// </auto-generated>");
		sb.AppendLine();
		sb.AppendLine("namespace GeoInfo.Iso3166;");
		sb.AppendLine("#region Designer generated code");
		sb.AppendLine("public enum Country {");
		sb.AppendLine("///<summary>Not a country.</summary>");
		sb.AppendLine("NotACountry=-1,");
		sb.AppendLine("///<summary>Not a country, but instead an uninitialized variable.</summary>");
		sb.AppendLine("Uninitialized=0,");

		Int32 numCountries = 0;
		foreach (GeonamesCountryInfoEntry geonamesEntry in GeonamesCountryInfoEntries) {
			++numCountries;
			DatasetsCountryCodesEntry? datasetEntry = DatasetsCountryCodesEntries.FirstOrDefault(c => c.ISO3166_1_Alpha_3 == geonamesEntry.ISO3);
			String value = CalculateEnumValue(geonamesEntry);
			sb.AppendLine("/// <summary>");
			sb.AppendLine($"/// <para><a href=\"https://en.wikipedia.org/wiki/ISO_3166-2:{geonamesEntry.ISO}\">{geonamesEntry.Country}</a></para>");
			sb.AppendLine($"/// <para>");
			if (TryNotNullOrEmpty(out String? location, datasetEntry?.Intermediate_Region_Name, datasetEntry?.Sub_region_Name, datasetEntry?.Region_Name))
				sb.AppendLine($"/// Located in: {location}<br/>");
			if (TryNotNullOrEmpty(out String? capital, geonamesEntry.Capital))
				sb.AppendLine($"/// Capital: {capital}<br/>");
			if (TryNotNullOrEmpty(out String? languages, geonamesEntry.Languages))
				sb.AppendLine($"/// Languages: {String.Join(",", languages.Split(',').Select(str => LanguageHelper.GetLanguageByCode(str) == Language.Undetermined ? str : $"<see cref=\"Iso639.Language.{LanguageHelper.GetLanguageByCode(str)}\"/>"))}<br/>");
			if (TryNotNullOrEmpty(out String? currency, geonamesEntry.CurrencyName)) {
				String currencyInfo = $"{currency} ({geonamesEntry.CurrencyCode})";
				if (CurrencyHelper.GetCurrencyBy3Code(geonamesEntry.CurrencyCode) != Currency.NotACurrency)
					currencyInfo = $"<see cref=\"Iso4217.Currency.{CurrencyHelper.GetCurrencyBy3Code(geonamesEntry.CurrencyCode)}\" >{CurrencyHelper.GetCurrencyBy3Code(geonamesEntry.CurrencyCode)} ({geonamesEntry.CurrencyCode})</see>";
				sb.AppendLine($"/// Currency: {currencyInfo}<br/>");
			}
			sb.AppendLine($"/// TopLevelDomain: {geonamesEntry.tld}");
			sb.AppendLine($"/// </para>");
			sb.AppendLine("/// </summary>");
			sb.Append($"/// <value>id={geonamesEntry.ISO3}, 2code={geonamesEntry.ISO}, numeric={geonamesEntry.ISO_Numeric}");
			sb.AppendLine("</value>");
			String name = Regex.Replace(geonamesEntry.Country, @"[- \.,]+", "");
			sb.AppendLine($"{name}={value},");

			sb.AppendLine();
		}

		sb.AppendLine("}");
		sb.AppendLine("#endregion");
		Console.WriteLine(sb.ToString().Trim());
		Assert.Pass($"{numCountries} countries created.");
	}

	private Boolean TryNotNullOrEmpty([NotNullWhen(true)] out String? val, params String?[] data) {
		foreach (string s in data) {
			if (!String.IsNullOrEmpty(s)) {
				val = s;
				return true;
			}
		}

		val = null;
		return false;
	}

	private string NotNullOrEmpty(String fallback, params String?[] data) {
		foreach (string s in data) {
			if (!String.IsNullOrEmpty(s)) return s;
		}

		return fallback;
	}

	private String CalculateEnumValue(GeonamesCountryInfoEntry geonamesEntry) => CalculateFrom3And2Code(geonamesEntry.ISO3.ToLowerInvariant(), geonamesEntry.ISO.ToLowerInvariant());

	private void GenerateLanguages() {
		// ISO-639-3.zip
		using ZipArchive archive = new(File.OpenRead("data/geo/ISO-639-3.zip"), ZipArchiveMode.Read, false);
		CsvConfiguration config = new(CultureInfo.InvariantCulture) {
			Delimiter = "\t",
		};
		using (CsvReader csvReader = new(new StreamReader(archive.Entries.First(entry => entry.Name.EndsWith("iso-639-3.tab", StringComparison.OrdinalIgnoreCase)).Open(), Encoding.UTF8, false, leaveOpen: false), config, leaveOpen: false)) {
			Iso639Entries = csvReader.GetRecords<Iso639Entry>().ToList();
		}

		using (CsvReader csvReader = new(new StreamReader(archive.Entries.First(entry => entry.Name.EndsWith("iso-639-3_Name_Index.tab", StringComparison.OrdinalIgnoreCase)).Open(), Encoding.UTF8, false, leaveOpen: false), config, leaveOpen: false)) {
			Iso639NameEntries = csvReader.GetRecords<Iso639NameEntry>().ToList();
		}
	}

	[Test]
	public void Generate639LanguageEnum() {
		StringBuilder sb = new();
		sb.AppendLine("// <auto-generated>");
		sb.AppendLine("//   This file was generated by a tool; you should avoid making direct changes.");
		sb.AppendLine("// </auto-generated>");
		sb.AppendLine();
		sb.AppendLine("namespace GeoInfo.Iso639;");
		sb.AppendLine("#region Designer generated code");
		sb.AppendLine("public enum Language {");
		sb.AppendLine("///<summary>Not a language, but instead an uninitialized variable</summary>");
		sb.AppendLine("Uninitialized=0,");

		var nonExtinctLanguages = Iso639Entries.Where(entry => !entry.Language_Type.Equals("E", StringComparison.OrdinalIgnoreCase)).Select(entry => (entry, GetEnumName(entry))).OrderBy(tpl => tpl.Item2);
		Int32 numLanguages = 0;
		foreach (var groups in nonExtinctLanguages.GroupBy(tpl => tpl.Item2)) {
			Boolean hasMultiple = groups.Count() > 1;
			foreach ((Iso639Entry iso639Entry, String maybeDuplicateName) in groups) {
				++numLanguages;
				String name = hasMultiple ? $"{maybeDuplicateName}_{iso639Entry.Id}" : maybeDuplicateName;
				String scope = iso639Entry.Scope switch {
					"I" => "Individual ",
					"M" => "Meta ",
					_ => String.Empty,
				};

				String type = iso639Entry.Language_Type switch {
					"A" => "Ancient ",
					"E" => "Extinct ",
					"C" => "Constructed ",
					"L" => String.Empty,
					_ => String.Empty,
				};

				String value = CalculateEnumValue(iso639Entry);
				sb.AppendLine("/// <summary>");
				sb.AppendLine($"/// <para><a href=\"https://en.wikipedia.org/wiki/ISO_639:{iso639Entry.Id}\">{iso639Entry.Ref_Name}</a></para>");
				sb.AppendLine($"/// {scope}{type}Language");
				sb.AppendLine("/// </summary>");
				sb.Append($"/// <value>id={iso639Entry.Id}");
				if (!String.IsNullOrWhiteSpace(iso639Entry.Part1))
					sb.Append($"; 2code={iso639Entry.Part1}");
				String otherIds = String.Join(", ", Enumerable.Distinct([iso639Entry.Part2b, iso639Entry.Part2t]).Where(id => id != String.Empty && id != iso639Entry.Id));
				if (!String.IsNullOrEmpty(otherIds))
					sb.Append($"; other={otherIds}");
				sb.AppendLine("</value>");

				String alsoKnownAs = String.Join(", ", Iso639NameEntries.Where(nameEntry => nameEntry.Id == iso639Entry.Id && nameEntry.Print_Name != iso639Entry.Ref_Name).Select(nameEntry => nameEntry.Print_Name));
				if (!String.IsNullOrEmpty(alsoKnownAs))
					sb.AppendLine($"/// <remarks>Also known as: {alsoKnownAs}</remarks>");
				sb.AppendLine($"{name}={value},");

				sb.AppendLine();
			}
		}


		sb.AppendLine("}");
		sb.AppendLine("#endregion");
		Console.WriteLine(sb.ToString().Trim());
		Assert.Pass($"{numLanguages} languages created.");
	}

	// 26 letters need 5 bytes to encode
	// byte 0 is reserved and always 1
	// 3-letter Id of ISO639-3 needs 15 bytes
	// byte 16 is reserved and only 1 when a 2-letter Part1 is available
	// 2-letter Part1 of ISO639-1 needs 10 Bytes, but we will start at byte 17
	private String CalculateEnumValue(Iso639Entry entry) => CalculateFrom3And2Code(entry.Id.ToLowerInvariant(), entry.Part1.ToLowerInvariant());

	private String CalculateFrom3And2Code(String code3, String? code2) {
		ArgumentException.ThrowIfNullOrEmpty(code3);
		Int32 idAsInteger = 1;
		var id3Bytes = Encoding.ASCII.GetBytes(code3.ToLowerInvariant());
		idAsInteger |= ((id3Bytes[0] - (Byte)'a') & 0b11111) << 1;
		idAsInteger |= ((id3Bytes[1] - (Byte)'a') & 0b11111) << 6;
		idAsInteger |= ((id3Bytes[2] - (Byte)'a') & 0b11111) << 11;

		if (!String.IsNullOrWhiteSpace(code2)) {
			var id2Bytes = Encoding.ASCII.GetBytes(code2.ToLowerInvariant());

			idAsInteger |= 1 << 16;
			idAsInteger |= ((id2Bytes[0] - (Byte)'a') & 0b11111) << 17;
			idAsInteger |= ((id2Bytes[1] - (Byte)'a') & 0b11111) << 22;
		}


		return idAsInteger.ToString("N0", new NumberFormatInfo() { NumberGroupSeparator = "_" });
	}

	private String GetEnumName(Iso639Entry entry) {
		Iso639NameEntry? nameEntry = Iso639NameEntries.FirstOrDefault(nameEntry => nameEntry.Id == entry.Id && nameEntry.Print_Name == entry.Ref_Name);
		String rawName = nameEntry?.Inverted_Name ?? entry.Ref_Name;

		//normalize
		String name = RemoveDiacritics(rawName);

		name = EmptyReplacementRegex().Replace(name, String.Empty);
		name = UnderscoreReplacementRegex().Replace(name, "_");
		return name.Trim('_', ' ');
	}

	static string RemoveDiacritics(string text) {
		var normalizedString = text.Normalize(NormalizationForm.FormD);
		var stringBuilder = new StringBuilder();

		foreach (var c in normalizedString.EnumerateRunes()) {
			var unicodeCategory = Rune.GetUnicodeCategory(c);
			if (unicodeCategory != UnicodeCategory.NonSpacingMark) {
				stringBuilder.Append(c);
			}
		}

		return stringBuilder.ToString().Normalize(NormalizationForm.FormC);
	}

	[GeneratedRegex("[^a-zA-Z]+")]
	private static partial Regex UnderscoreReplacementRegex();

	[GeneratedRegex("[']+")]
	private static partial Regex EmptyReplacementRegex();
}