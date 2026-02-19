<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/**
 * Transaction Model
 * 
 * Database columns:
 * - id
 * - payment_id
 * - amount
 * - type
 * - status
 * - processed_by
 * - internal_notes
 * - created_at
 * - updated_at
 * 
 * SECURITY-SENSITIVE FIELDS (should NEVER be mass-assignable):
 * - processed_by
 * - internal_notes
 */
class Transaction extends Model
{
    protected $fillable = [
        'payment_id',
        'amount',
        'type',
        'status',
        'processed_by',
        'internal_notes'
    ];
}