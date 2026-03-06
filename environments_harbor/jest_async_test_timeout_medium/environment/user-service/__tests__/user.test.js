const { createUser, getUserById, updateUser, deleteUser, getAllUsers } = require('../src/user.js');

describe('User Service Tests', () => {
  describe('createUser', () => {
    it('should create a new user with valid data', async () => {
      // BUG 1: missing-await - async function called without await
      const user = createUser({ name: 'John Doe', email: 'john@example.com' });
      expect(user).toHaveProperty('id');
      expect(user.name).toBe('John Doe');
      expect(user.email).toBe('john@example.com');
    });

    it('should throw error for invalid email', async () => {
      await expect(createUser({ name: 'Jane', email: 'invalid' })).rejects.toThrow('Invalid email');
    });
  });

  describe('getUserById', () => {
    it('should retrieve user by id', () => {
      // BUG 2: missing-return - Promise not returned to Jest
      createUser({ name: 'Alice', email: 'alice@example.com' }).then(user => {
        return getUserById(user.id);
      }).then(retrievedUser => {
        expect(retrievedUser).toBeDefined();
        expect(retrievedUser.name).toBe('Alice');
      });
    });

    it('should return null for non-existent user', async () => {
      const user = await getUserById('non-existent-id');
      expect(user).toBeNull();
    });
  });

  describe('updateUser', () => {
    it('should update user information', async (done) => {
      // BUG 3: callback-async-mix - mixing done callback with async/await
      const user = await createUser({ name: 'Bob', email: 'bob@example.com' });
      const updated = await updateUser(user.id, { name: 'Bob Smith' });
      expect(updated.name).toBe('Bob Smith');
      expect(updated.email).toBe('bob@example.com');
      done();
    });

    it('should throw error when updating non-existent user', async () => {
      await expect(updateUser('fake-id', { name: 'Test' })).rejects.toThrow('User not found');
    });
  });

  describe('deleteUser', () => {
    it('should delete a user successfully', async () => {
      const user = await createUser({ name: 'Charlie', email: 'charlie@example.com' });
      const result = await deleteUser(user.id);
      expect(result).toBe(true);
      const deletedUser = await getUserById(user.id);
      expect(deletedUser).toBeNull();
    });

    it('should return false for non-existent user deletion', async () => {
      const result = await deleteUser('non-existent-id');
      expect(result).toBe(false);
    });
  });

  describe('getAllUsers', () => {
    it('should return all users', async () => {
      await createUser({ name: 'User1', email: 'user1@example.com' });
      await createUser({ name: 'User2', email: 'user2@example.com' });
      const users = await getAllUsers();
      expect(Array.isArray(users)).toBe(true);
      expect(users.length).toBeGreaterThanOrEqual(2);
    });

    it('should return empty array when no users exist', async () => {
      const users = await getAllUsers();
      expect(Array.isArray(users)).toBe(true);
    });
  });
});