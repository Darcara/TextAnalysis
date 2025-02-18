namespace TextAnalysis.Test.Generators;

using System.Diagnostics.CodeAnalysis;
using System.Xml.Serialization;

[SuppressMessage("ReSharper", "InconsistentNaming")]
public sealed record Iso639Entry(String Id, String Part2b, String Part2t, String Part1, String Scope, String Language_Type, String Ref_Name, String Comment);

[SuppressMessage("ReSharper", "InconsistentNaming")]
public sealed record Iso639NameEntry(String Id, String Print_Name, String Inverted_Name);

public sealed record DatasetsCountryCodesEntry {
	public String FIFA;
	public String Dial;
	public String ISO3166_1_Alpha_3;
	public String MARC;
	public String is_independent;
	public String ISO3166_1_numeric;
	public String GAUL;
	public String FIPS;
	public String WMO;
	public String ISO3166_1_Alpha_2;
	public String ITU;
	public String IOC;
	public String DS;
	public String UNTERM_Spanish_Formal;
	public String Global_Code;
	public String Intermediate_Region_Code;
	public String official_name_fr;
	public String UNTERM_French_Short;
	public String ISO4217_currency_name;
	public String UNTERM_Russian_Formal;
	public String UNTERM_English_Short;
	public String ISO4217_currency_alphabetic_code;
	public String Small_Island_Developing_States_SIDS;
	public String UNTERM_Spanish_Short;
	public String ISO4217_currency_numeric_code;
	public String UNTERM_Chinese_Formal;
	public String UNTERM_French_Formal;
	public String UNTERM_Russian_Short;
	public String M49;
	public String Sub_region_Code;
	public String Region_Code;
	public String official_name_ar;
	public String ISO4217_currency_minor_unit;
	public String UNTERM_Arabic_Formal;
	public String UNTERM_Chinese_Short;
	public String Land_Locked_Developing_Countries_LLDC;
	public String Intermediate_Region_Name;
	public String official_name_es;
	public String UNTERM_English_Formal;
	public String official_name_cn;
	public String official_name_en;
	public String ISO4217_currency_country_name;
	public String Least_Developed_Countries_LDC;
	public String Region_Name;
	public String UNTERM_Arabic_Short;
	public String Sub_region_Name;
	public String official_name_ru;
	public String Global_Name;
	public String Capital;
	public String Continent;
	public String TLD;
	public String Languages;
	public String Geoname_ID;
	public String CLDR_display_name;
	public String EDGAR;
	public String wikidata_id;
}

public sealed record GeonamesCountryInfoEntry {
	public String ISO;
	public String ISO3;
	public String ISO_Numeric;
	public String fips;
	public String Country;
	public String Capital;
	public String Area_in_sq_km;
	public String Population;
	public String Continent;
	public String tld;
	public String CurrencyCode;
	public String CurrencyName;
	public String Phone;
	public String Postal_Code_Format;
	public String Postal_Code_Regex;
	public String Languages;
	public String geonameid;
	public String neighbours;
	public String EquivalentFipsCode;
}

[XmlRoot(ElementName="CcyNtry")]
public class CurrencyEntry { 

	[XmlElement(ElementName="CcyNm")] 
	public CcyNm CcyNm; 

	[XmlElement(ElementName="Ccy")] 
	public string? Ccy; 

	[XmlElement(ElementName="CcyNbr")] 
	public int CcyNbr; 

	[XmlElement(ElementName="CcyMnrUnts")] 
	public string CcyMnrUnts; 

	[XmlElement(ElementName="CtryNm")] 
	public string CtryNm; 
}

[XmlRoot(ElementName="CcyNm")]
public class CcyNm { 

	[XmlAttribute(AttributeName="IsFund")] 
	public bool IsFund; 

	[XmlText] 
	public string Text; 
}

[XmlRoot(ElementName="CcyTbl")]
public class CcyTbl { 

	[XmlElement(ElementName="CcyNtry")] 
	public List<CurrencyEntry> CcyNtry; 
}

[XmlRoot(ElementName="ISO_4217")]
public class ISO4217 { 

	[XmlElement(ElementName="CcyTbl")] 
	public CcyTbl CcyTbl; 

	[XmlAttribute(AttributeName="Pblshd")] 
	public DateTime Pblshd; 

	[XmlText] 
	public string Text; 
}

