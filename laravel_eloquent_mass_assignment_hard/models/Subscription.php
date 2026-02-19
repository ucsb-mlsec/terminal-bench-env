<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/**
 * Subscription Model
 * 
 * Database columns:
 * - id
 * - user_id
 * - plan_id
 * - status
 * - start_date
 * - end_date
 * - created_at
 * - updated_at
 * 
 * Security-sensitive fields (should never be mass-assignable):
 * - id
 * - created_at
 * - updated_at
 */
class Subscription extends Model
{
    /**
     * The attributes that are not mass assignable.
     *
     * @var array
     */
    protected $guarded = ['*'];
}