require 'active_record'

class Comment < ActiveRecord::Base
  belongs_to :post
  belongs_to :user
end