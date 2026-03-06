const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const UserSchema = new Schema({
  username: String,
  email: String,
  password: String,
  firstName: String,
  lastName: String,
  age: Number,
  phone: String,
  role: String,
  status: String,
  bio: String,
  website: String,
  profileImage: String,
  tags: [String],
  preferences: {
    emailNotifications: Boolean,
    theme: String,
    language: String
  },
  createdAt: Date,
  lastLogin: Date
});

module.exports = mongoose.model('User', UserSchema);