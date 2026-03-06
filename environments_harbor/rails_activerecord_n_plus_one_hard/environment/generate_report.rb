require 'active_record'
require 'sqlite3'
require_relative 'post'
require_relative 'comment'
require_relative 'author'

ActiveRecord::Base.establish_connection(
  adapter: 'sqlite3',
  database: './blog.db'
)

ActiveRecord::Base.logger = Logger.new(STDOUT)

query_count = 0
ActiveSupport::Notifications.subscribe('sql.active_record') do |name, start, finish, id, payload|
  unless payload[:name] == 'SCHEMA'
    query_count += 1
  end
end

posts = Post.all

posts.each do |post|
  puts "Post: #{post.title}"
  puts "Author: #{post.author.name}"
  puts "Comments: #{post.comments.count}"
  
  post.comments.each do |comment|
    puts "- #{comment.body} by #{comment.author.name}"
  end
  
  puts ""
end

puts "Total queries executed: #{query_count}"