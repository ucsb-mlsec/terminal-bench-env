<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/**
 * Invoice Model
 * 
 * Database columns:
 * - id
 * - customer_id
 * - amount
 * - status
 * - due_date
 * - created_at
 * - updated_at
 * 
 * Security-sensitive fields: id, created_at, updated_at
 */
class Invoice extends Model
{
    /**
     * The attributes that are mass assignable.
     *
     * @var array
     */
    protected $fillable = ['customer_id', 'amount', 'status', 'due_date'];

    /**
     * The attributes that aren't mass assignable.
     *
     * @var array
     */
    protected $guarded = ['id', 'created_at', 'updated_at'];
}