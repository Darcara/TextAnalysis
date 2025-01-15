Enums for ISO languages, countries and currencies

# ISO639 Language

```csharp
Language l = Language.English;
// Every Language has a 639-3 3 code
String iso639_3Code = Language.English.Get3Code();  // eng
String iso639_2Code = Language.English.Get2Code();  // en

Language english = LanguageHelper.GetLanguageBy2Code("en");
Language english = LanguageHelper.GetLanguageBy2Code("en-us");
Language english = LanguageHelper.GetLanguageBy3Code("eng");
```

# ISO3166 Countries
# ISO427 Currency