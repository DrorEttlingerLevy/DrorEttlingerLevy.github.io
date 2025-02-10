require 'tzinfo'

begin
  TZInfo::DataSource.set(:ruby)
  puts "✅ Jekyll is using Ruby timezone data!"
rescue TZInfo::DataSourceNotFound, TZInfo::DataSources::InvalidZoneinfoDirectory => e
  puts "❌ ERROR: #{e.message}"
end
