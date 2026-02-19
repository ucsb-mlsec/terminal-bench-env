<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/**
 * Payment Model
 * 
 * Database columns:
 * - id
 * - invoice_id
 * - amount
 * - payment_method
 * - description
 * - internal_reference
 * - admin_notes
 * - created_at
 * - updated_at
 * 
 * Security-sensitive fields (should NOT be mass-assignable):
 * - internal_reference
 * - admin_notes
 */
class Payment extends Model
{
    /**
     * The attributes that are mass assignable.
     *
     * @var array
     */
    protected $fillable = [
        'amount',
        'payment_method',
    ];
}