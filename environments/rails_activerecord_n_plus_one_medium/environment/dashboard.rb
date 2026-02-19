require 'active_record'
require 'sqlite3'
require_relative 'models/post'
require_relative 'models/user'
require_relative 'models/comment'

ActiveRecord::Base.establish_connection(
  adapter: 'sqlite3',
  database: 'blog.db'
)

ActiveRecord::Base.logger = Logger.new(STDOUT)

puts "Dashboard - Recent Posts"
puts "=" * 80

posts = Post.order(created_at: :desc).limit(20)

posts.each do |post|
  author_name = post.user.name
  comment_count = post.comments.count
  
  puts "Post: #{post.title} | Author: #{author_name} | Comments: #{comment_count}"
end

puts "=" * 80
puts "Displayed #{posts.length} posts"