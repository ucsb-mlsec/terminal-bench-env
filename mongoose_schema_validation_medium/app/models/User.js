const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  username: { type: String },
  email: { type: String },
  age: { type: Number },
  phoneNumber: { type: String },
  website: { type: String },
  bio: { type: String },
  role: { type: String },
  accountStatus: { type: String },
  createdAt: { type: Date }
});

module.exports = mongoose.model('User', UserSchema);