require 'active_record'

class User < ActiveRecord::Base
  has_many :posts
  has_many :comments
end