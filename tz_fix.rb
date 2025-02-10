require 'tzinfo'

begin
  # Force tzinfo to use the built-in Ruby data source
  TZInfo::DataSource.set(:ruby)
  puts "✅ TZInfo data source set to Ruby mode!"
rescue TZInfo::DataSourceNotFound, TZInfo::DataSources::InvalidZoneinfoDirectory => e
  puts "❌ ERROR: #{e.message}"
end
