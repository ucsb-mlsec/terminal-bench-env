#!/usr/bin/env python3

import unittest
import asyncio
from tortoise import Tortoise
from tortoise.exceptions import DoesNotExist
import sys
import os

# Add the payment_app directory to the path
sys.path.insert(0, '/workspace/payment_app')

from models import User
from transfer import transfer_funds


class TestTransferFunds(unittest.TestCase):
    
    async def asyncSetUp(self):
        """Initialize database and create test users"""
        await Tortoise.init(
            db_url='sqlite://:memory:',
            modules={'models': ['models']}
        )
        await Tortoise.generate_schemas()
        
        # Create test users
        self.alice = await User.create(name='Alice', balance=1000.0)
        self.bob = await User.create(name='Bob', balance=500.0)
    
    async def asyncTearDown(self):
        """Close database connections"""
        await Tortoise.close_connections()
    
    def setUp(self):
        """Sync wrapper for async setUp"""
        asyncio.run(self.asyncSetUp())
    
    def tearDown(self):
        """Sync wrapper for async tearDown"""
        asyncio.run(self.asyncTearDown())
    
    async def async_test_successful_transfer(self):
        """Test that a valid transfer updates both accounts correctly"""
        # Get initial balances
        alice_initial = self.alice.balance
        bob_initial = self.bob.balance
        
        # Perform transfer
        result = await transfer_funds(self.alice.id, self.bob.id, 200.0)
        
        # Verify transfer succeeded
        self.assertTrue(result)
        
        # Reload users from database to verify actual state
        await self.alice.refresh_from_db()
        await self.bob.refresh_from_db()
        
        # Verify balances updated correctly
        self.assertEqual(self.alice.balance, alice_initial - 200.0)
        self.assertEqual(self.bob.balance, bob_initial + 200.0)
        self.assertEqual(self.alice.balance, 800.0)
        self.assertEqual(self.bob.balance, 700.0)
    
    def test_successful_transfer(self):
        """Sync wrapper for successful transfer test"""
        asyncio.run(self.async_test_successful_transfer())
    
    async def async_test_insufficient_funds(self):
        """Test that insufficient funds prevents any changes"""
        # Get initial balances
        alice_initial = self.alice.balance
        bob_initial = self.bob.balance
        
        # Try to transfer more than available
        result = await transfer_funds(self.alice.id, self.bob.id, 2000.0)
        
        # Verify transfer failed
        self.assertFalse(result)
        
        # Reload users from database
        await self.alice.refresh_from_db()
        await self.bob.refresh_from_db()
        
        # Verify NO changes to either account
        self.assertEqual(self.alice.balance, alice_initial)
        self.assertEqual(self.bob.balance, bob_initial)
        self.assertEqual(self.alice.balance, 1000.0)
        self.assertEqual(self.bob.balance, 500.0)
    
    def test_insufficient_funds(self):
        """Sync wrapper for insufficient funds test"""
        asyncio.run(self.async_test_insufficient_funds())
    
    async def async_test_invalid_from_user(self):
        """Test that invalid from_user raises exception and makes no changes"""
        # Get initial balances
        bob_initial = self.bob.balance
        
        # Try to transfer from non-existent user
        with self.assertRaises(DoesNotExist):
            await transfer_funds(9999, self.bob.id, 100.0)
        
        # Reload Bob from database
        await self.bob.refresh_from_db()
        
        # Verify Bob's balance unchanged
        self.assertEqual(self.bob.balance, bob_initial)
        self.assertEqual(self.bob.balance, 500.0)
    
    def test_invalid_from_user(self):
        """Sync wrapper for invalid from_user test"""
        asyncio.run(self.async_test_invalid_from_user())
    
    async def async_test_invalid_to_user(self):
        """Test that invalid to_user raises exception and makes no changes"""
        # Get initial balances
        alice_initial = self.alice.balance
        
        # Try to transfer to non-existent user
        with self.assertRaises(DoesNotExist):
            await transfer_funds(self.alice.id, 9999, 100.0)
        
        # Reload Alice from database
        await self.alice.refresh_from_db()
        
        # Verify Alice's balance unchanged
        self.assertEqual(self.alice.balance, alice_initial)
        self.assertEqual(self.alice.balance, 1000.0)
    
    def test_invalid_to_user(self):
        """Sync wrapper for invalid to_user test"""
        asyncio.run(self.async_test_invalid_to_user())
    
    async def async_test_negative_amount(self):
        """Test that negative amounts are rejected with no changes"""
        # Get initial balances
        alice_initial = self.alice.balance
        bob_initial = self.bob.balance
        
        # Try to transfer negative amount
        result = await transfer_funds(self.alice.id, self.bob.id, -100.0)
        
        # Verify transfer failed
        self.assertFalse(result)
        
        # Reload users from database
        await self.alice.refresh_from_db()
        await self.bob.refresh_from_db()
        
        # Verify NO changes to either account
        self.assertEqual(self.alice.balance, alice_initial)
        self.assertEqual(self.bob.balance, bob_initial)
    
    def test_negative_amount(self):
        """Sync wrapper for negative amount test"""
        asyncio.run(self.async_test_negative_amount())
    
    async def async_test_zero_amount(self):
        """Test that zero amounts are rejected with no changes"""
        # Get initial balances
        alice_initial = self.alice.balance
        bob_initial = self.bob.balance
        
        # Try to transfer zero amount
        result = await transfer_funds(self.alice.id, self.bob.id, 0.0)
        
        # Verify transfer failed
        self.assertFalse(result)
        
        # Reload users from database
        await self.alice.refresh_from_db()
        await self.bob.refresh_from_db()
        
        # Verify NO changes to either account
        self.assertEqual(self.alice.balance, alice_initial)
        self.assertEqual(self.bob.balance, bob_initial)
    
    def test_zero_amount(self):
        """Sync wrapper for zero amount test"""
        asyncio.run(self.async_test_zero_amount())
    
    async def async_test_rollback_on_error(self):
        """Test that mid-transfer errors roll back all changes"""
        # Get initial balances
        alice_initial = self.alice.balance
        bob_initial = self.bob.balance
        
        # This test verifies that if anything goes wrong during transfer,
        # the database is left unchanged. We test with an invalid recipient.
        try:
            await transfer_funds(self.alice.id, 9999, 100.0)
        except DoesNotExist:
            pass  # Expected exception
        
        # Reload users from database
        await self.alice.refresh_from_db()
        await self.bob.refresh_from_db()
        
        # Verify NO changes to any account (complete rollback)
        self.assertEqual(self.alice.balance, alice_initial)
        self.assertEqual(self.bob.balance, bob_initial)
        self.assertEqual(self.alice.balance, 1000.0)
    
    def test_rollback_on_error(self):
        """Sync wrapper for rollback test"""
        asyncio.run(self.async_test_rollback_on_error())
    
    async def async_test_multiple_transfers(self):
        """Test multiple sequential transfers maintain consistency"""
        # Transfer 100 from Alice to Bob
        result1 = await transfer_funds(self.alice.id, self.bob.id, 100.0)
        self.assertTrue(result1)
        
        # Transfer 50 from Bob to Alice
        result2 = await transfer_funds(self.bob.id, self.alice.id, 50.0)
        self.assertTrue(result2)
        
        # Reload users
        await self.alice.refresh_from_db()
        await self.bob.refresh_from_db()
        
        # Verify final balances: Alice: 1000 - 100 + 50 = 950, Bob: 500 + 100 - 50 = 550
        self.assertEqual(self.alice.balance, 950.0)
        self.assertEqual(self.bob.balance, 550.0)
    
    def test_multiple_transfers(self):
        """Sync wrapper for multiple transfers test"""
        asyncio.run(self.async_test_multiple_transfers())


if __name__ == '__main__':
    unittest.main()